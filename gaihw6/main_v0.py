import os
import glob
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


# ==========================================
# 1. Dataset 讀取與預處理
# ==========================================
class BrainrotDataset(Dataset):
    def __init__(self, img_dir, img_size=64):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.*"))
        self.img_paths = [
            p
            for p in self.img_paths
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


# ==========================================
# 2. DD-GAN 組件：時間編碼、生成器與判別器
# ==========================================
class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block that injects time+latent embedding into features.

    This block preserves spatial resolution unless `down=True` (applies stride-2 conv).
    The embedding is projected to `out_channels` and broadcast-added to the feature map.
    A learnable 1x1 conv adapts the skip connection when shapes differ.
    """

    def __init__(self, in_channels, out_channels, embed_dim, down=False):
        super().__init__()
        self.down = down
        stride = 2 if down else 1
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.GroupNorm(max(1, out_channels // 16), out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(max(1, out_channels // 16), out_channels)

        self.embed_proj = nn.Linear(embed_dim, out_channels)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        # emb: (B, embed_dim)
        h = self.conv1(x)
        # project embedding and add
        emb_proj = self.embed_proj(emb).view(emb.size(0), -1, 1, 1)
        h = h + emb_proj
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return h + self.skip(x)


class Generator(nn.Module):
    """U-Net style generator with skip connections and embedding injection into every block."""

    def __init__(self, img_size=64, c_in=3, c_out=3, latent_dim=128, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.GELU()
        )
        self.z_mlp = nn.Sequential(nn.Linear(latent_dim, time_dim), nn.GELU())

        # Downsampling path: produce and save intermediate features for skip connections
        self.down1 = ResidualBlock(
            c_in, 64, embed_dim=time_dim, down=False
        )  # 64x64 -> 64x64
        self.down2 = ResidualBlock(
            64, 128, embed_dim=time_dim, down=True
        )  # 64x64 -> 32x32
        self.down3 = ResidualBlock(
            128, 256, embed_dim=time_dim, down=True
        )  # 32x32 -> 16x16

        # Bottleneck (no spatial change)
        self.bottleneck = ResidualBlock(256, 256, embed_dim=time_dim, down=False)

        # Up-sampling: use ConvTranspose to upsample, then concatenate skip and apply ResidualBlock
        self.up_trans1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 32x32
        self.up_block1 = ResidualBlock(
            128 + 128, 128, embed_dim=time_dim, down=False
        )  # concat with down2 (128)

        self.up_trans2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 32x32 -> 64x64
        self.up_block2 = ResidualBlock(
            64 + 64, 64, embed_dim=time_dim, down=False
        )  # concat with down1 (64)

        # Final conv to produce output image
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, c_out, kernel_size=3, padding=1), nn.Tanh()
        )

    def forward(self, x_t, t, z):
        # prepare embedding
        t_emb = self.time_mlp(t)
        z_emb = self.z_mlp(z)
        emb = t_emb + z_emb  # (B, time_dim)

        # Encoder -- save skips
        d1 = self.down1(x_t, emb)  # B x 64 x 64 x 64
        d2 = self.down2(d1, emb)  # B x 128 x 32 x 32
        d3 = self.down3(d2, emb)  # B x 256 x 16 x 16

        # Bottleneck
        b = self.bottleneck(d3, emb)  # B x 256 x 16 x 16

        # Upsample + concat with corresponding skip, then residual block
        u = self.up_trans1(b)  # B x 128 x 32 x 32
        u = torch.cat([u, d2], dim=1)  # B x 256 x 32 x 32
        u = self.up_block1(u, emb)  # B x 128 x 32 x 32

        u = self.up_trans2(u)  # B x 64 x 64 x 64
        u = torch.cat([u, d1], dim=1)  # B x 128 x 64 x 64
        u = self.up_block2(u, emb)  # B x 64 x 64 x 64

        out = self.final_conv(u)
        return out


class Discriminator(nn.Module):
    """【已加深】多層條件式判別器 (下採樣至 8x8)"""

    def __init__(self, img_size=64, c_in=3, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.GELU()
        )

        # 加深的特徵提取層
        self.net = nn.Sequential(
            nn.Conv2d(c_in * 2, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.time_dense = nn.Linear(time_dim, 256)
        # PatchGAN: output a spatial grid of patch scores (8x8) instead of a single scalar
        self.output_layer = nn.Conv2d(
            256, 1, kernel_size=1, stride=1, padding=0
        )  # 8x8 -> 8x8

    def forward(self, x_pred_or_real, x_t, t):
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x_pred_or_real, x_t], dim=1)
        h = self.net(x_input)
        h = h + self.time_dense(t_emb).view(h.size(0), h.size(1), 1, 1)
        # Output shape: (B, 1, 8, 8) for patch-based discrimination
        return self.output_layer(h)


# ==========================================
# 2.5 零卷積 (Zero-Convolution) 插入機制
# ==========================================
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlledGenerator(nn.Module):
    def __init__(self, trained_generator):
        super().__init__()
        self.base_gen = trained_generator
        for param in self.base_gen.parameters():
            param.requires_grad = False
        # only need a small zero-conv to inject control signal into input image
        self.zero_conv_in = ZeroConv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x_t, t, z, control_signal):
        # inject control into input and delegate to base generator
        controlled_input = x_t + self.zero_conv_in(control_signal)
        return self.base_gen(controlled_input, t, z)


# ==========================================
# 3. DD-GAN 主類別與訓練流程
# ==========================================
class DDGAN(nn.Module):
    def __init__(self, img_size=64, latent_dim=128, num_timesteps=4):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.T = num_timesteps

        self.gen = Generator(img_size=img_size, latent_dim=latent_dim)
        self.disc = Discriminator(img_size=img_size)

        self.register_buffer("beta", torch.linspace(0.02, 0.2, self.T))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_hat", torch.cumprod(self.alpha, dim=0))

        # ========== EMA (Exponential Moving Average) for Generator ==========
        self.ema_decay = 0.999
        self.gen_shadow_state = {}
        for name, param in self.gen.named_parameters():
            if param.requires_grad:
                self.gen_shadow_state[name] = param.data.clone()
        self.gen_original_state = {}

    def update_ema(self):
        """Update EMA shadow weights after Generator optimization step."""
        for name, param in self.gen.named_parameters():
            if param.requires_grad and name in self.gen_shadow_state:
                self.gen_shadow_state[name] = (
                    self.ema_decay * self.gen_shadow_state[name]
                    + (1.0 - self.ema_decay) * param.data
                )

    def apply_ema_weights(self):
        """Temporarily swap Generator weights with EMA shadow weights."""
        self.gen_original_state = {}
        for name, param in self.gen.named_parameters():
            if param.requires_grad and name in self.gen_shadow_state:
                self.gen_original_state[name] = param.data.clone()
                param.data = self.gen_shadow_state[name].clone()

    def restore_gen_weights(self):
        """Restore Generator weights from backup."""
        for name, param in self.gen.named_parameters():
            if param.requires_grad and name in self.gen_original_state:
                param.data = self.gen_original_state[name]

    def to_device_ema(self, device):
        """Move EMA shadow state to the specified device."""
        for name in self.gen_shadow_state:
            self.gen_shadow_state[name] = self.gen_shadow_state[name].to(device)

    def q_sample(self, x_0, t, noise):
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise

    def train_gan(self, dataloader, epochs=10, lr=2e-4, device="cuda"):
        self.to(device)
        self.to_device_ema(device)
        opt_G = torch.optim.AdamW(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = torch.optim.AdamW(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))
        # Use LSGAN (Least Squares GAN) for more stable training
        criterion = nn.MSELoss()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:

            epoch_task = progress.add_task("[cyan]總訓練進度", total=epochs)

            for epoch in range(epochs):
                batch_task = progress.add_task(
                    f"[yellow]Epoch {epoch+1}/{epochs}", total=len(dataloader)
                )

                for x_0 in dataloader:
                    x_0 = x_0.to(device)
                    batch_size = x_0.size(0)

                    t = torch.randint(0, self.T, (batch_size,), device=device)
                    noise = torch.randn_like(x_0)
                    x_t = self.q_sample(x_0, t, noise)
                    latent_z = torch.randn(batch_size, self.latent_dim, device=device)

                    # ----------------------------
                    # (1) 訓練判別器 Discriminator (PatchGAN)
                    # ----------------------------
                    opt_D.zero_grad()
                    pred_real = self.disc(x_0, x_t, t)  # (B, 1, 8, 8)
                    loss_D_real = criterion(pred_real, torch.ones_like(pred_real))

                    x_0_fake = self.gen(x_t, t, latent_z)
                    pred_fake = self.disc(x_0_fake.detach(), x_t, t)  # (B, 1, 8, 8)
                    loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

                    # LSGAN discriminator loss: average of real and fake MSE (works with patch grid)
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)
                    loss_D.backward()
                    opt_D.step()

                    # ----------------------------
                    # (2) 訓練生成器 Generator (with EMA update)
                    # ----------------------------
                    opt_G.zero_grad()
                    pred_fake_G = self.disc(x_0_fake, x_t, t)  # (B, 1, 8, 8)
                    # LSGAN generator adversarial loss pushes discriminator output to 1
                    loss_G_adv = criterion(pred_fake_G, torch.ones_like(pred_fake_G))
                    loss_G_recon = nn.functional.l1_loss(x_0_fake, x_0)

                    loss_G = loss_G_adv + 1.0 * loss_G_recon
                    loss_G.backward()
                    opt_G.step()

                    # Update EMA weights after Generator optimization
                    self.update_ema()

                    progress.update(
                        batch_task,
                        advance=1,
                        description=f"D_Loss: {loss_D.item():.4f} | G_Loss: {loss_G.item():.4f}",
                    )

                progress.remove_task(batch_task)
                progress.update(epoch_task, advance=1)

    @torch.no_grad()
    def generate_save_and_evaluate_fid(
        self,
        dataloader,
        num_images=2000,
        batch_size=32,
        output_dir="generated_images/",
        device="cuda",
    ):
        """【一體化優化】生成 2000 張圖、命名 0001.png~2000.png 存檔，並直接用這批圖計算 FID 分數
        Uses EMA weights for improved generation quality and FID score.
        """
        from torchmetrics.image.fid import FrechetInceptionDistance

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[FID & 生成評估] 正在初始化 InceptionV3 模型...")
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        # ensure model and buffers are on the correct device
        self.to(device)

        # Apply EMA weights for generation
        print(f"[FID & 生成評估] 正在應用 EMA 權重進行生成...")
        self.apply_ema_weights()
        self.gen.eval()

        print(
            f"[FID & 生成評估] 正在生成並儲存 {num_images} 張融合獸圖片 (規範命名 0001.png ~ 2000.png)..."
        )
        num_batches = math.ceil(num_images / batch_size)
        saved_count = 0

        try:
            for b in range(num_batches):
                current_batch_size = min(batch_size, num_images - saved_count)
                if current_batch_size <= 0:
                    break

                # 反向擴散生成
                x_t = torch.randn(
                    current_batch_size, 3, self.img_size, self.img_size, device=device
                )
                for t_idx in reversed(range(self.T)):
                    t_vec = torch.full(
                        (current_batch_size,), t_idx, dtype=torch.long, device=device
                    )
                    z = torch.randn(current_batch_size, self.latent_dim, device=device)
                    x_0_pred = self.gen(x_t, t_vec, z)
                    if t_idx > 0:
                        alpha_t = self.alpha[t_idx].to(device)
                        x_t = (
                            torch.sqrt(alpha_t) * x_t
                            + (1 - alpha_t) * x_0_pred
                            + 0.01 * torch.randn_like(x_t)
                        )
                    else:
                        x_t = x_0_pred

                # 還原至 [0, 1] 範圍
                normalized_imgs = (x_t + 1) / 2
                normalized_imgs = torch.clamp(normalized_imgs, 0, 1)

                # 逐張儲存，並使用 4 位數補零命名 (0001.png, 0002.png ...)
                for i in range(current_batch_size):
                    saved_count += 1
                    img_tensor = normalized_imgs[i].cpu()
                    img = transforms.ToPILImage()(img_tensor)
                    img.save(os.path.join(output_dir, f"{saved_count:04d}.png"))

                # 將剛才生成好且存檔的這批 tensor 轉成 uint8 直接更新到 FID 假圖池
                fake_imgs_uint8 = (normalized_imgs * 255).to(torch.uint8)
                fid_metric.update(fake_imgs_uint8, real=False)

            print(f"成功將 {saved_count} 張圖片儲存至 {output_dir}")

            # 讀取真實訓練集圖片特徵
            print(f"[FID & 生成評估] 正在讀取真實訓練集圖片特徵...")
            for x_0 in dataloader:
                x_0 = x_0.to(device)
                real_imgs = (x_0 + 1) / 2
                real_imgs = torch.clamp(real_imgs, 0, 1)
                real_imgs = (real_imgs * 255).to(torch.uint8)

                fid_metric.update(real_imgs, real=True)

            print(f"[FID & 生成評估] 正在計算最終 Fréchet 距離...")
            fid_score = fid_metric.compute().item()
        finally:
            # Always restore original weights even if generation fails
            self.restore_gen_weights()
            print(f"[FID & 生成評估] 已恢復原始 Generator 權重")

        return fid_score


# ==========================================
# 4. 主執行 Pipeline
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 配置
    IMG_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 100

    # 1. 讀取 Dataset
    train_dir = os.path.join("dataset", "trainset")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        print(f"提示：請將 4799 張圖片放入 {train_dir} 資料夾中。")

    dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # 2. 初始化 DD-GAN
    ddgan_model = DDGAN(img_size=IMG_SIZE, latent_dim=128, num_timesteps=4)

    # 3. 執行訓練 (內含 Rich 進度條)
    if len(dataset) > 0:
        print(f"開始訓練【加深版】DD-GAN 模型 (共 {EPOCHS} 輪)...")
        ddgan_model.train_gan(dataloader, epochs=EPOCHS, device=device)

        # 儲存基礎模型權重
        torch.save(ddgan_model.state_dict(), "ddgan_base.pth")
        print("基礎模型已儲存為 ddgan_base.pth")

        # 4 & 5. 🚀 執行 2000 張生成存檔並同步計算 FID Score
        output_folder = "generated_images/"
        fid_result = ddgan_model.generate_save_and_evaluate_fid(
            dataloader,
            num_images=2000,
            batch_size=BATCH_SIZE,
            output_dir=output_folder,
            device=device,
        )
        print(
            f"\n🔥 訓練完成！這 2000 張已存檔圖片對比 Trainset 的 FID Score 為: {fid_result:.4f}"
        )

        # ==========================================
        # 擴充：訓練完成後，插入 Zero-Convolution
        # ==========================================
        print("\n--- 正在為已訓練的模型插入 Zero-Convolution 架構 ---")
        controlled_gen = ControlledGenerator(ddgan_model.gen)
        controlled_gen.to(device)

        # 儲存帶有 Zero-Conv 的新架構模型
        torch.save(controlled_gen.state_dict(), "ddgan_with_zeroconv.pth")
        print("成功建立並儲存帶有 Zero-Convolution 的新模型：ddgan_with_zeroconv.pth")
        print(
            "此時 base_gen 的權重已被凍結，您可以開始針對 trainable_down 與 zero_conv 進行條件微調！"
        )
