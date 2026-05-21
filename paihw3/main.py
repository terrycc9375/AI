import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# =====================================================================
# 0. 環境與可重複性設定 (Global Configurations)
# =====================================================================
def set_seed(seed=42):
    """
    設定隨機種子以確保實驗結果的可重複性。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =====================================================================
# 1. 2D 瑞士捲資料集生成 (Data Provider)
# =====================================================================
def generate_swiss_roll(n_samples=3000, noise_std=0.05):
    """
    生成 2D Swiss Roll Toy 資料集並進行標準化。
    
    參數:
    - n_samples: 生成的點雲總數
    - noise_std: 原始資料自帶的雜訊厚度
    
    返回:
    - torch.Tensor: 形狀為 (n_samples, 2) 且經標準化的點雲
    """
    # 均勻採樣 theta 區間，控制螺旋繞行的圈數
    theta = np.linspace(1.5 * np.pi, 4.5 * np.pi, n_samples)
    a = 0.4
    
    # 計算Swiss Roll Toy坐標(x,y)
    x = a * theta * np.cos(theta)
    y = a * theta * np.sin(theta)
    x0 = np.stack([x, y], axis=1)
    
    # 加入高斯噪音
    noise = np.random.normal(0, noise_std, size=x0.shape)
    x0 = x0 + noise
    
    # 預處理：標準化至 [-2, 2] 區間，穩定擴散模型的訓練尺度
    x0_mean = np.mean(x0, axis=0)
    x0_std = np.std(x0, axis=0)
    x0_normalized = (x0 - x0_mean) / x0_std
    
    return torch.tensor(x0_normalized, dtype=torch.float32)


# =====================================================================
# 2. 逆向去噪網路組件 (Model Components)
# =====================================================================
class SinusoidalEmbedding(nn.Module):
    """
    時間步的 Positional Encoding 模組。
    把一個整數的 timestep t，轉換成一組 128 維的滑順三角函數向量，
    讓去噪的 MLP 大腦能辨識現在是在哪一個擴散階段。
    """
    def __init__(self, dim):
        super().__init__()
        # dim 是最終要輸出的總維度（例如 128）
        self.dim = dim

    def forward(self, t):
        device = t.device
        # 因為後面要把 sin 和 cos 拼起來，所以這裡先除以 2 算出各自的維度（例如 64）
        half_dim = self.dim // 2
        # 依據 Transformer 論文公式，計算出 64 個不同波段的頻率基底
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # 利用 Broadcasting 機制，讓每個時間步 t 去跟這 64 個不同的頻率做相乘
        # t[:, None] 的形狀變為 (Batch, 1)
        # emb[None, :] 的形狀變為 (1, 64)
        # 相乘後得到矩陣形狀為 (Batch, 64)，代表每個樣本在 64 個頻率下的角速度(相位)
        emb = t[:, None] * emb[None, :]

        # 分別對這 64 個維度計算 sin 與 cos 值，再把兩者在最後一個維度(dim=-1)拼接起來
        # 拼接後形狀： (Batch, 64) + (Batch, 64) -> (Batch, 128)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class DenoisingMLP(nn.Module):
    """
    用於 2D 點雲去噪的多層感知機 (MLP) 主幹網路。
    輸入當前帶噪點雲 x_t 與時間步 t，輸出預測的噪聲 epsilon。
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 時間步嵌入層
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 空間特徵提取層
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2)  # 輸出預測的 2D 雜訊
        
        self.act = nn.SiLU()
    
    def forward(self, xt, t):
        # 1. 計算時間特徵向量
        t_emb = self.time_mlp(t)  # (Batch_size, hidden_dim)
        
        # 2. 融入空間特徵
        h1 = self.act(self.fc1(xt))
        h = h1 + t_emb  # 此時維度為 hidden_dim (128)
        
        # 3. 深度特徵演進（修正後的殘差連接）
        # 先升維到 256 再降維回 128
        h_deep = self.act(self.fc2(h))
        h_deep = self.act(self.fc3(h_deep))
        
        # 在這裡進行相加！因為 h 和 h_deep 的維度此時都是 128，可以完美對齊
        h = h + h_deep  
        
        # 4. 預測噪聲
        return self.fc4(h)


# =====================================================================
# 3. DDPM 核心演算系統 (Core Engine Class)
# =====================================================================
class DDPM:
    """
    封裝了擴散模型完整生命週期的演算系統。
    包含：前向加噪、模型訓練、以及逆向採樣生成。
    """
    def __init__(self, T=300, hidden_dim=1024, device=DEVICE):
        self.T = T
        self.device = device
        
        # 設定 Linear Beta 排程 (從 1e-4 到 0.015)
        self.betas = torch.linspace(1e-4, 0.015, T).to(self.device)
        self.alphas = 1.0 - self.betas
        # 計算重要的累積乘積 \bar{\alpha}_t
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)
        
        # 實例化去噪神經網路
        self.model = DenoisingMLP(hidden_dim=hidden_dim).to(self.device)
        self.loss_history = []

    def forward_diffusion(self, x0, t):
        """
        擴散前向加噪過程 q(x_t | x_0)。利用再參數化技巧一步到位計算 x_t。
        """
        alpha_bar_t = self.alphas_bar[t].view(-1, 1)
        epsilon = torch.randn_like(x0).to(self.device)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon
        return xt, epsilon

    def train(self, dataset, batch_size=256, epochs=80, lr=1e-3):
        """
        執行逆向去噪網路的訓練主迴圈，並記錄 MSE 損失。
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Train DDPM with T = {self.T}")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x0 in train_loader:
                batch_x0 = batch_x0.to(self.device)
                b_size = batch_x0.shape[0]
                
                # 隨機抽樣時間步
                t = torch.randint(0, self.T, (b_size,), device=self.device).long()
                
                # 前向加噪
                xt, epsilon_true = self.forward_diffusion(batch_x0, t)
                
                # 預測噪聲並更新網路
                epsilon_pred = self.model(xt, t)
                loss = nn.functional.mse_loss(epsilon_pred, epsilon_true)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * b_size
                
            avg_loss = epoch_loss / len(dataset)
            self.loss_history.append(avg_loss)
            
            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch [{epoch+1}/{epochs}] - MSE Loss: {avg_loss:.6f}")

    def plot_loss_curve(self, save_path="ddpm_loss_curve.png"):
        """繪製並儲存訓練損失下降曲線 (Fig.2 要求)"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, label='Training Loss', color='tab:orange', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'DDPM Training Loss Curve (T={self.T})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(save_path)
        #plt.show()

    def save_weights(self, path="ddpm_swiss_roll.pth"):
        """儲存模型權重"""
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path="ddpm_swiss_roll.pth"):
        """載入模型權重"""
        self.model.load_weights(torch.load(path, map_location=self.device))


    @torch.no_grad()
    def p_sample_loop(self, n_samples=3000):
        """
        從純高斯雜訊開始，一路逆向迭代 T 步，最終還原出 2D 瑞士捲。
        並記錄特定時間步的點雲，用來繪製採樣軌跡。
        """
        self.model.eval()
        
        # 1. 初始化：從標準高斯分佈中隨機抽樣 3000 個點作為起點 (t = T)
        xt = torch.randn(n_samples, 2, device=self.device)
        
        # 建立一個字典，用來存下特定時間步的點雲位置，方便後面畫軌跡
        trajectory = {}
        
        # 動態挑選要記錄並展示的 8 個時間步 (包含起點、中途與終點)
        # 例如 T=300 時會挑出 [299, 256, 213, 170, 128, 85, 42, 0]
        steps_to_save = np.linspace(self.T - 1, 0, 8, dtype=int)
        
        # print(f"正在執行逆向採樣軌跡生成 (從 t = {self.T-1} 倒扣到 0)...")
        
        # 2. 從最後一步 T-1 開始，倒退迭代回到 0
        for t_val in reversed(range(self.T)):
            # 建立當前時間步的張量 (Batch_size,)
            t_tensor = torch.full((n_samples,), t_val, dtype=torch.long, device=self.device)
            
            # 呼叫神經網路預測當前時間步的雜訊
            eps_pred = self.model(xt, t_tensor)
            
            # 提取當前步的預計算排程係數
            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alphas_bar[t_val]
            beta_t = self.betas[t_val]
            
            # 如果 t_val > 0，則抽樣隨機擾動 z；若到了最後一步 t_val == 0，則不加擾動
            z = torch.randn_like(xt) if t_val > 0 else 0.0
            sigma_t = torch.sqrt(beta_t) # 或者使用 \sqrt{(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t) * \beta_t}
            
            # 執行 DDPM 逆向採樣核心公式
            xt = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred) + sigma_t * z
            
            # 如果是我們指定的展示步，就把點雲存起來
            if t_val in steps_to_save:
                trajectory[t_val] = xt.cpu().numpy()
                
        return trajectory, steps_to_save

    @torch.no_grad()
    def sample(self, n_samples=3000):
        """Generate final DDPM samples from pure noise."""
        self.model.eval()
        xt = torch.randn(n_samples, 2, device=self.device)
        for t_val in reversed(range(self.T)):
            t_tensor = torch.full((n_samples,), t_val, dtype=torch.long, device=self.device)
            eps_pred = self.model(xt, t_tensor)
            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alphas_bar[t_val]
            beta_t = self.betas[t_val]
            z = torch.randn_like(xt) if t_val > 0 else 0.0
            sigma_t = torch.sqrt(beta_t)
            xt = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred) + sigma_t * z
        return xt.cpu().numpy()


class BasicGAN:
    """A simple GAN for 2D swiss roll generation."""
    def __init__(self, latent_dim=16, hidden_dim=128, device=DEVICE, lr=2e-4, betas=(0.5, 0.999)):
        self.device = device
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2)
        ).to(self.device)
        self.discriminator = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr * 3, betas=betas)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_history = {'gen': [], 'disc': []}

    def train(self, dataset, batch_size=256, epochs=200, record_n=8, record_path="gan_swiss_roll.png"):
        """Train GAN with real swiss roll points and record snapshots.

        record_n: number of snapshots to collect (including epoch 1 and final epoch)
        record_path: output path for composite image containing the snapshots
        """
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        real_label = 0.9
        fake_label = 0.0

        record_epochs = set(np.linspace(1, epochs, record_n, dtype=int))
        snapshots = {}

        with Progress(
            TextColumn("[#238ce8][progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[#faac2f]Fake Rate: {task.fields[test_acc]:>5.2f}%"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task("Training Progress", total=epochs, test_acc=0.0)
            for epoch in range(1, epochs + 1):
                self.generator.train()
                self.discriminator.train()
                gen_loss_epoch = 0.0
                disc_loss_epoch = 0.0
                fooled = 0
                total_fake = 0

                for real_data in train_loader:
                    real_data = real_data.to(self.device)
                    b_size = real_data.size(0)
                    label_real = torch.full((b_size, 1), real_label, device=self.device)
                    label_fake = torch.full((b_size, 1), fake_label, device=self.device)
                    noise = torch.randn(b_size, self.latent_dim, device=self.device)
                    fake_data = self.generator(noise)

                    self.dis_optimizer.zero_grad()
                    output_real = self.discriminator(real_data)
                    loss_real = self.criterion(output_real, label_real)
                    output_fake = self.discriminator(fake_data.detach())
                    loss_fake = self.criterion(output_fake, label_fake)
                    loss_disc = loss_real + loss_fake
                    loss_disc.backward()
                    self.dis_optimizer.step()

                    self.gen_optimizer.zero_grad()
                    output_fake_for_gen = self.discriminator(fake_data)
                    loss_gen = self.criterion(output_fake_for_gen, label_real)
                    loss_gen.backward()
                    self.gen_optimizer.step()

                    gen_loss_epoch += loss_gen.item() * b_size
                    disc_loss_epoch += loss_disc.item() * b_size
                    # discriminator outputs logits when using BCEWithLogitsLoss; use sigmoid to get probabilities
                    fooled += (torch.sigmoid(output_fake) >= 0.5).sum().item()
                    total_fake += b_size

                self.loss_history['gen'].append(gen_loss_epoch / len(dataset))
                self.loss_history['disc'].append(disc_loss_epoch / len(dataset))
                fake_rate = fooled / total_fake if total_fake > 0 else 0.0
                progress.update(
                    main_task,
                    advance=1,
                    description=f"Epoch {epoch}/{epochs}",
                    test_acc=fake_rate * 100,
                )

                if epoch in record_epochs:
                    snapshots[epoch] = self.sample(n_samples=3000)

        # After training, compose snapshots into a single figure and save
        if len(snapshots) > 0:
            sorted_epochs = sorted(snapshots.keys())
            cols = 4
            rows = int(np.ceil(len(sorted_epochs) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()
            for i, e in enumerate(sorted_epochs):
                pts = snapshots[e]
                axes[i].scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.6, color='tab:orange')
                axes[i].set_title(f"epoch={e}")
                axes[i].set_xlim(-2.5, 2.5)
                axes[i].set_ylim(-2.5, 2.5)
                axes[i].set_aspect('equal')
            # hide any unused axes
            for j in range(i + 1, rows * cols):
                axes[j].axis('off')
            plt.tight_layout()
            plt.savefig(record_path)

    def save_model(self, path="gan_swiss_roll.pth"):
        """Save generator and discriminator state."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)

    def load_model(self, path="gan_swiss_roll.pth"):
        """Load generator and discriminator state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

    def sample(self, n_samples=3000):
        """Generate swiss roll samples from random latent vectors."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(noise).cpu().numpy()
        self.generator.train()
        return samples

    def plot_loss_curves(self, save_path="basic_gan_loss_curves.png"):
        """Plot and save generator and discriminator loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.loss_history['gen']) + 1)
        ax.plot(epochs, self.loss_history['gen'], label='Generator Loss', color='tab:blue', linewidth=2)
        ax.plot(epochs, self.loss_history['disc'], label='Discriminator Loss', color='tab:orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('BasicGAN Training Loss Curves')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class AdvancedGAN:
    """GAN with hinge loss."""
    def __init__(self, latent_dim=16, hidden_dim=128, device=DEVICE, lr=1e-4, betas=(0.5, 0.999)):
        self.device = device
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2)
        ).to(self.device)
        self.discriminator = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr * 3, betas=betas)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.loss_history = {'gen': [], 'disc': []}

    def train(self, dataset, batch_size=256, epochs=200, record_n=8, record_path="advanced_gan_swiss_roll.png"):
        """Train GAN using hinge loss."""
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        record_epochs = set(np.linspace(1, epochs, record_n, dtype=int))
        snapshots = {}

        with Progress(
            TextColumn("[#238ce8][progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[#faac2f]Fake Rate: {task.fields[test_acc]:>5.2f}%"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task("Training Progress", total=epochs, test_acc=0.0)
            for epoch in range(1, epochs + 1):
                self.generator.train()
                self.discriminator.train()
                gen_loss_epoch = 0.0
                disc_loss_epoch = 0.0
                fooled = 0
                total_fake = 0

                for real_data in train_loader:
                    real_data = real_data.to(self.device)
                    b_size = real_data.size(0)
                    noise = torch.randn(b_size, self.latent_dim, device=self.device)
                    fake_data = self.generator(noise)

                    # Discriminator hinge loss
                    self.dis_optimizer.zero_grad()
                    output_real = self.discriminator(real_data)
                    output_fake = self.discriminator(fake_data.detach())
                    loss_real = torch.mean(torch.relu(1.0 - output_real))
                    loss_fake = torch.mean(torch.relu(1.0 + output_fake))
                    loss_disc = loss_real + loss_fake
                    loss_disc.backward()
                    self.dis_optimizer.step()

                    # Generator hinge loss
                    self.gen_optimizer.zero_grad()
                    output_fake_for_gen = self.discriminator(fake_data)
                    loss_gen = -torch.mean(output_fake_for_gen)
                    loss_gen.backward()
                    self.gen_optimizer.step()

                    gen_loss_epoch += loss_gen.item() * b_size
                    disc_loss_epoch += loss_disc.item() * b_size
                    fooled += (output_fake >= 0.0).sum().item()
                    total_fake += b_size

                self.loss_history['gen'].append(gen_loss_epoch / len(dataset))
                self.loss_history['disc'].append(disc_loss_epoch / len(dataset))
                fake_rate = fooled / total_fake if total_fake > 0 else 0.0
                progress.update(
                    main_task,
                    advance=1,
                    description=f"Epoch {epoch}/{epochs}",
                    test_acc=fake_rate * 100,
                )

                if epoch in record_epochs:
                    snapshots[epoch] = self.sample(n_samples=3000)

        if len(snapshots) > 0:
            sorted_epochs = sorted(snapshots.keys())
            cols = 4
            rows = int(np.ceil(len(sorted_epochs) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()
            for i, e in enumerate(sorted_epochs):
                pts = snapshots[e]
                axes[i].scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.6, color='tab:orange')
                axes[i].set_title(f"epoch={e}")
                axes[i].set_xlim(-2.5, 2.5)
                axes[i].set_ylim(-2.5, 2.5)
                axes[i].set_aspect('equal')
            for j in range(i + 1, rows * cols):
                axes[j].axis('off')
            plt.tight_layout()
            plt.savefig(record_path)

    def save_model(self, path="advanced_gan_swiss_roll.pth"):
        """Save generator and discriminator state."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)

    def load_model(self, path="advanced_gan_swiss_roll.pth"):
        """Load generator and discriminator state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

    def sample(self, n_samples=3000):
        """Generate swiss roll samples from random latent vectors."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(noise).cpu().numpy()
        self.generator.train()
        return samples

    def plot_loss_curves(self, save_path="advanced_gan_loss_curves.png"):
        """Plot and save generator and discriminator loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.loss_history['gen']) + 1)
        ax.plot(epochs, self.loss_history['gen'], label='Generator Loss', color='tab:blue', linewidth=2)
        ax.plot(epochs, self.loss_history['disc'], label='Discriminator Loss', color='tab:orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('AdvancedGAN Training Loss Curves')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def gaussian_kernel(x, y, sigma=1.0):
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.t() - 2.0 * x @ y.t()
    return torch.exp(-dist / (2.0 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """Compute the RBF-kernel MMD between two point clouds."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)
    mmd = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return mmd.item()


def chamfer_distance(x, y):
    """Compute symmetric Chamfer distance between two point clouds."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    dxy = torch.cdist(x, y, p=2)
    dist_x = dxy.min(dim=1).values
    dist_y = dxy.min(dim=0).values
    return (dist_x.mean() + dist_y.mean()).item()


def plot_reverse_trajectory(trajectory, steps_to_save, T):
    """
    將逆向採樣的軌跡繪製成 2x4 的子圖 (對應作業 Fig.2 要求)。
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    
    # 按照時間從大到小（從雜訊到原圖）排序並畫圖
    sorted_steps = sorted(steps_to_save, reverse=True)
    
    for i, t_val in enumerate(sorted_steps):
        pts = trajectory[t_val]
        
        # 標題特殊處理：第一步叫 noise，最後一步叫 final
        if i == 0:
            title_str = f"noise (t={t_val})"
        elif i == 7:
            title_str = "final"
        else:
            title_str = f"t = {t_val}"
            
        axes[i].scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.6, color='tab:green')
        axes[i].set_title(title_str)
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(-2.5, 2.5)
        axes[i].set_aspect('equal')
        
    plt.suptitle(f"DDPM reverse sampling trajectory [T={T}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"reverse_trajectory_T{T}.png")
    #plt.show()


# =====================================================================
# 4. 前向與逆向採樣實驗展示 (Visualization Functions)
# =====================================================================
def run_forward_diffusion_demo(ddpm_instance, dataset):
    """
    自動根據目前的 T 分配時間步，繪製前向加噪點雲圖 (Fig.1 要求)。
    此函式能自動適應 T=300 或 T=30。
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    
    # 動態產生要展示的 8 個時間步
    timesteps_to_show = np.linspace(0, ddpm_instance.T - 1, 8, dtype=int)
    x0_all = dataset.to(ddpm_instance.device)
    
    for i, t_val in enumerate(timesteps_to_show):
        if t_val == 0:
            xt = x0_all
        else:
            t_tensor = torch.full((x0_all.shape[0],), t_val, dtype=torch.long).to(ddpm_instance.device)
            xt, _ = ddpm_instance.forward_diffusion(x0_all, t_tensor)
            
        xt_np = xt.cpu().numpy()
        axes[i].scatter(xt_np[:, 0], xt_np[:, 1], s=2, alpha=0.6, color='tab:blue')
        axes[i].set_title(f"t = {t_val}")
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(-2.5, 2.5)
        axes[i].set_aspect('equal')
        
    plt.suptitle(f"Forward diffusion q(x_t | x_0) [T={ddpm_instance.T}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"forward_diffusion_T{ddpm_instance.T}.png")
    #plt.show()


# =====================================================================
# 5. 主執行進入點 (Main Entry Point)
# =====================================================================
def p1():
    """
    Part 1
    """
    print("=== [Part 1.1] ===")
    swiss_data = generate_swiss_roll(n_samples=3000, noise_std=0.05)
    ddpm_system = DDPM(T=300, hidden_dim=1024, device=DEVICE)
    run_forward_diffusion_demo(ddpm_system, swiss_data)
    ddpm_system.train(swiss_data, batch_size=64, epochs=200, lr=1e-3)
    ddpm_system.plot_loss_curve(save_path=f"ddpm_loss_curve_T{ddpm_system.T}.png")
    ddpm_system.save_weights(path=f"ddpm_weights_T{ddpm_system.T}.pth")

    print("=== [Part 1.2] ===")
    ddpm_variant = DDPM(T=50 , hidden_dim=1024, device=DEVICE)
    run_forward_diffusion_demo(ddpm_variant, swiss_data)
    ddpm_variant.train(swiss_data, batch_size=64, epochs=200, lr=1e-3)
    ddpm_variant.plot_loss_curve(save_path=f"ddpm_loss_curve_T{ddpm_variant.T}.png")
    traj_var, steps_var = ddpm_variant.p_sample_loop(n_samples=3000)
    plot_reverse_trajectory(traj_var, steps_var, ddpm_variant.T)

    ddpm_variant = DDPM(T=1000, hidden_dim=1024, device=DEVICE)
    run_forward_diffusion_demo(ddpm_variant, swiss_data)
    ddpm_variant.train(swiss_data, batch_size=64, epochs=200, lr=1e-3)
    ddpm_variant.plot_loss_curve(save_path=f"ddpm_loss_curve_T{ddpm_variant.T}.png")
    traj_var, steps_var = ddpm_variant.p_sample_loop(n_samples=3000)
    plot_reverse_trajectory(traj_var, steps_var, ddpm_variant.T)

    print("=== [Part 1.3] ===")
    traj, steps = ddpm_system.p_sample_loop(n_samples=3000)
    plot_reverse_trajectory(traj, steps, ddpm_system.T)

def p2():
    """Train DDPM and GANs, then compare all models using MMD."""
    swiss_data = generate_swiss_roll(n_samples=3000, noise_std=0.05)

    print("=== Training DDPM ===")
    ddpm = DDPM(T=300, hidden_dim=1024, device=DEVICE)
    ddpm.train(swiss_data, batch_size=64, epochs=200, lr=1e-3)
    ddpm.save_weights(path="ddpm_weights.pth")
    fake_ddpm = ddpm.sample(n_samples=3000)

    print("=== Training BasicGAN ===")
    gan = BasicGAN(latent_dim=64, hidden_dim=256, device=DEVICE, lr=1e-4)
    gan.train(swiss_data, batch_size=256, epochs=300)
    gan.plot_loss_curves()
    gan.save_model(path="basic_gan.pth")
    fake_basic = gan.sample(n_samples=3000)

    print("=== Training AdvancedGAN ===")
    gan2 = AdvancedGAN(latent_dim=32, hidden_dim=256, device=DEVICE, lr=1e-4)
    gan2.train(swiss_data, batch_size=256, epochs=300)
    gan2.plot_loss_curves()
    gan2.save_model(path="advanced_gan.pth")
    fake_advanced = gan2.sample(n_samples=3000)

    real_samples = swiss_data.numpy()
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    print("\n=== MMD results for multiple sigma values ===")
    for sigma in sigma_values:
        mmd_ddpm = compute_mmd(real_samples, fake_ddpm, sigma=sigma)
        mmd_basic = compute_mmd(real_samples, fake_basic, sigma=sigma)
        mmd_advanced = compute_mmd(real_samples, fake_advanced, sigma=sigma)
        print(f"sigma={sigma:>3.1f} | DDPM: {mmd_ddpm:.6f} | BasicGAN: {mmd_basic:.6f} | AdvancedGAN: {mmd_advanced:.6f}")

    mmd_ddpm = compute_mmd(real_samples, fake_ddpm, sigma=1.0)
    mmd_basic = compute_mmd(real_samples, fake_basic, sigma=1.0)
    mmd_advanced = compute_mmd(real_samples, fake_advanced, sigma=1.0)

    print("\n=== Summary at sigma=1.0 ===")
    print(f"DDPM MMD: {mmd_ddpm:.6f}")
    print(f"BasicGAN MMD: {mmd_basic:.6f}")
    print(f"AdvancedGAN MMD: {mmd_advanced:.6f}")
    best_model = min([('DDPM', mmd_ddpm), ('BasicGAN', mmd_basic), ('AdvancedGAN', mmd_advanced)], key=lambda item: item[1])
    print(f"Best model by MMD at sigma=1.0: {best_model[0]} (MMD={best_model[1]:.6f})")

    print("\n=== Chamfer distance (DDPM vs AdvancedGAN) ===")
    chamfer_ddpm = chamfer_distance(real_samples, fake_ddpm)
    chamfer_advanced = chamfer_distance(real_samples, fake_advanced)
    print(f"DDPM Chamfer: {chamfer_ddpm:.6f}")
    print(f"AdvancedGAN Chamfer: {chamfer_advanced:.6f}")
    if chamfer_ddpm < chamfer_advanced:
        print("DDPM is better by Chamfer distance.")
    elif chamfer_advanced < chamfer_ddpm:
        print("AdvancedGAN is better by Chamfer distance.")
    else:
        print("DDPM and AdvancedGAN have the same Chamfer distance.")


if __name__ == "__main__":
    p2()