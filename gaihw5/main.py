import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import DDPMScheduler, AutoencoderKL
import torch.nn as nn
import math


# --- Simple U-Net implementation (no pretrained components) ---
def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # add time embedding
        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, channel_mults=(1, 2, 4, 8), num_res_blocks=2, out_channels=4, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # down blocks
        self.downs = nn.ModuleList()
        in_ch = base_channels
        down_channels = [in_ch]
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = nn.ModuleList([ResBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim) for i in range(num_res_blocks)])
            downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            
            self.downs.append(nn.ModuleDict({
                "blocks": blocks,
                "downsample": downsample
            }))
            in_ch = out_ch
            down_channels.append(in_ch)

        # middle
        self.mid1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid2 = ResBlock(in_ch, in_ch, time_emb_dim)

        # up blocks
        self.ups = nn.ModuleList()
        up_mults = list(reversed(channel_mults[:-1])) + [channel_mults[0]]
        
        for i, mult in enumerate(up_mults):
            out_ch = base_channels * mult
            upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
            
            skip_ch = down_channels[-(i + 1)]
            blocks = nn.ModuleList([
                ResBlock(out_ch + skip_ch if j == 0 else out_ch, out_ch, time_emb_dim)
                for j in range(num_res_blocks)
            ])
            
            self.ups.append(nn.ModuleDict({
                "upsample": upsample,
                "blocks": blocks
            }))
            in_ch = out_ch

        self.final_norm = nn.GroupNorm(8, in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, timesteps):
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        t_emb = timestep_embedding(timesteps, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        hs = []
        h = self.init_conv(x)

        # down
        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, t_emb)
            hs.append(h)
            h = down["downsample"](h)

        # middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # up
        for up in self.ups:
            h = up["upsample"](h)
            skip = hs.pop()
            
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
                
            h = torch.cat([h, skip], dim=1)
            for block in up["blocks"]:
                h = block(h, t_emb)

        h = self.final_norm(h)
        h = self.final_act(h)
        return self.final_conv(h)


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = sorted([p for p in self.image_dir.glob("*.png") if p.is_file()])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image


@torch.no_grad()
def generate_and_save_images(unet, vae, scheduler, epoch, device, save_folder):
    """訓練中的定期採樣監聽（維持不變，每次產4張）"""
    unet.eval()
    n_samples = 4
    x = torch.randn(n_samples, 4, 32, 32, device=device)

    ddim_steps = 50 
    timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=device)

    betas = scheduler.betas.clone().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_alpha_prev(t_idx, timesteps):
        if t_idx == len(timesteps) - 1:
            return torch.tensor(1.0, device=device)
        return alphas_cumprod[timesteps[t_idx + 1]]

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        pred_noise = unet(x, t_tensor)
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_prev = get_alpha_prev(i, timesteps)

        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        direction_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_xt

    decoded = vae.decode(x / vae.config.scaling_factor).sample
    decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
    for i in range(decoded.shape[0]):
        out_path = os.path.join(save_folder, f"sample_{epoch}_{i}.png")
        save_image(decoded[i], out_path)
        
    unet.train()


# ========== 新增：訓練完成後的 3000 張大批量加速生成函數 ==========
@torch.no_grad()
def generate_final_results(unet, vae, scheduler, device, results_folder, total_images=3000, batch_size=50):
    """
    使用 DDIM 在訓練完成後快速批量生成指定數量的圖片。
    """
    unet.eval()
    os.makedirs(results_folder, exist_ok=True)
    print(f"\n正在開始生成 {total_images} 張圖片至 {results_folder} 資料夾...")

    # 配置 DDIM
    ddim_steps = 50
    timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=device)
    betas = scheduler.betas.clone().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_alpha_prev(t_idx, timesteps):
        if t_idx == len(timesteps) - 1:
            return torch.tensor(1.0, device=device)
        return alphas_cumprod[timesteps[t_idx + 1]]

    generated_count = 0
    
    # 計算需要跑幾個大 batch
    num_batches = math.ceil(total_images / batch_size)
    
    for b in tqdm(range(num_batches), desc="Generating Images"):
        # 最後一個 batch 可能不滿 batch_size
        current_batch_size = min(batch_size, total_images - generated_count)
        
        # 1. 從標準正態分佈初始化潛在空間噪聲
        x = torch.randn(current_batch_size, 4, 32, 32, device=device)
        
        # 2. DDIM 採樣循環
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
            pred_noise = unet(x, t_tensor)
            
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_prev = get_alpha_prev(i, timesteps)
            
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            direction_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_xt
            
        # 3. 解碼並儲存
        decoded = vae.decode(x / vae.config.scaling_factor).sample
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
        
        for i in range(decoded.shape[0]):
            generated_count += 1
            # 使用四位數流水號命名（例如：0001.png, 0325.png）
            out_path = os.path.join(results_folder, f"{generated_count:04d}.png")
            save_image(decoded[i], out_path)

    print(f"成功完成！所有圖片已儲存至 {results_folder}")


def train():
    # ========= Hyperparameters ==========
    train_epochs = 50 
    batch_size = 8 
    gradient_accumulation_steps = 1 
    lr = 1e-4
    eval_freq = 1000 
    image_dir = "public_data/images"
    output_dir = "outputs"
    results_dir = "results" # 新增：最終生成的目標資料夾
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Saved folders ==========
    save_folder = os.path.join(output_dir, "samples")
    os.makedirs(save_folder, exist_ok=True)

    # ========== Load Pretrained Model ==========
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    # ========== Init ==========
    unet = UNet(
        in_channels=4,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        out_channels=4,
        time_emb_dim=256,
    ).to(device)
    unet.train()
    optimizer = torch.optim.Adam(list(unet.parameters()), lr=lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)

    # ========== Dataset ==========
    dataset = ImageDataset(image_dir, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # ========== Training ==========
    loss_accumulated = 0.0
    step = 0
    pbar = tqdm(total=train_epochs * len(dataloader), desc="Training")
    for epoch in range(train_epochs):
        for batch in dataloader:
            step += 1
            pixel_values = batch.to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            betas = noise_scheduler.betas.to(device)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            T = betas.shape[0]

            timesteps = torch.randint(0, T, (latents.shape[0],), device=device).long()

            noise = torch.randn_like(latents)
            a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_1_a_t = torch.sqrt(1.0 - a_t)
            noisy_latents = sqrt_a_t * latents + sqrt_1_a_t * noise

            pred = unet(noisy_latents, timesteps)
            if isinstance(pred, torch.Tensor):
                pred_noise = pred
            elif hasattr(pred, 'sample'):
                pred_noise = pred.sample
            else:
                pred_noise = pred[0]

            loss = F.mse_loss(pred_noise, noise)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss_accumulated += loss.item() * gradient_accumulation_steps
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})

        if step % eval_freq == 0:
            generate_and_save_images(unet, vae, noise_scheduler, step, device, save_folder)
            unet.train()

    # 儲存最後的模型權重
    torch.save(unet.state_dict(), os.path.join(save_folder, "unet.pt"))
    
    # ========== 訓練完成：呼叫 3000 張生成任務 ==========
    # 這裡調整 batch_size=50（可依據你的 GPU 顯存大小改成 30 或 100）
    generate_final_results(
        unet=unet, 
        vae=vae, 
        scheduler=noise_scheduler, 
        device=device, 
        results_folder=results_dir, 
        total_images=3000, 
        batch_size=50
    )

if __name__ == "__main__":
    train()