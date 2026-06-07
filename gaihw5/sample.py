import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import DDPMScheduler, AutoencoderKL
from tqdm import tqdm
from PIL import Image

# =====================================================================
# --- 1. 保持與訓練完全一致的自定義 UNet 架構 ---
# =====================================================================
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

        self.mid1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid2 = ResBlock(in_ch, in_ch, time_emb_dim)

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

        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, t_emb)
            hs.append(h)
            h = down["downsample"](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

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


# =====================================================================
# --- 2. 核心推論與 DDIM 快速採樣函數 ---
# =====================================================================
def generate_and_save_images(unet, vae, scheduler, device, save_folder, num_samples, batch_size=32):
    unet.eval()
    vae.eval()

    os.makedirs(save_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()

    total_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0

    # 預先計算 DDIM 50 步排程所需的數值
    ddim_steps = 50
    timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=device)
    
    betas = scheduler.betas.clone().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_alpha_prev(t_idx, timesteps):
        if t_idx == len(timesteps) - 1:
            return torch.tensor(1.0, device=device)
        return alphas_cumprod[timesteps[t_idx + 1]]

    print(f"開始生成 {num_samples} 張圖片 (使用 DDIM {ddim_steps} 步加速)...")

    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches), desc="Generating Batches"):
            current_batch = min(batch_size, num_samples - sample_idx)
            
            # 從高斯噪聲初始化 Latent (4, 32, 32)
            latents = torch.randn(current_batch, 4, 32, 32, device=device)

            # DDIM 採樣循環
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((current_batch,), t, device=device, dtype=torch.long)
                
                # 用自定義 UNet 預測噪聲
                pred_noise = unet(latents, t_tensor)
                
                alpha_bar_t = alphas_cumprod[t]
                alpha_bar_prev = get_alpha_prev(i, timesteps)
                
                # DDIM 更新公式 (eta=0, 確定性取樣)
                pred_x0 = (latents - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                direction_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
                latents = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_xt

            # 縮放並解碼 Latents
            latents = latents / vae.config.scaling_factor
            images = vae.decode(latents, return_dict=False)[0]
            images = (images.clamp(-1, 1) + 1) / 2

            # 將批次內影像存為檔案
            for i in range(current_batch):
                output_path = os.path.join(save_folder, f"{sample_idx:04d}.png")
                image = to_pil(images[i].cpu())
                image.save(output_path)
                sample_idx += 1

    print(f"成功將 {sample_idx} 張生成圖片儲存至 {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained custom UNet model.")
    # 這裡調整為讀取你的權重檔案路徑，例如 unet.pt
    parser.add_argument("--checkpoint_path", type=str, default="./outputs/samples/unet.pt",
                        help="Path to the trained custom UNet state_dict file (e.g., unet.pt).")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory where generated images will be saved.")
    parser.add_argument("--num_samples", type=int, default=3000,
                        help="Total number of images to generate.")
    # 建議設為 16 或 32。如果再度跳出 cuDNN 錯誤，請將此處改為 16 或 8
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for latent generation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading VAE and Custom UNet on {device}")

    # 載入 VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    vae.eval()

    # 初始化你自己的 UNet 實例
    unet = UNet(
        in_channels=4,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        out_channels=4,
        time_emb_dim=256,
    ).to(device)
    
    # 載入訓練好的權重
    if os.path.exists(args.checkpoint_path):
        unet.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint from {args.checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}, running with random weights!")
        
    unet.requires_grad_(False)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)

    generate_and_save_images(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        device=device,
        save_folder=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )