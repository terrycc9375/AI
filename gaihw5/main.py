import math
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler


# =====================================================================
# --- 1. 時間編碼基礎模組 ---
# =====================================================================
class Timesteps(nn.Module):

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)

        if self.flip_sin_to_cos:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        else:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedding(nn.Module):

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# =====================================================================
# --- 2. 核心架構組件 (AdaGN ResNetBlock 與 Self-Attention) ---
# =====================================================================
class ImprovedResnetBlock2D(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, temb_channels: int, groups: int = 32
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-5)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(temb_channels, out_channels * 2)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-5)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, hidden_states, temb):
        h = self.norm1(hidden_states)
        h = self.act1(h)
        h = self.conv1(h)

        time_emb = self.time_emb_proj(temb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(time_emb, 2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.conv_shortcut(hidden_states)


class AttentionBlock2D(nn.Module):

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, eps=1e-5)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        b, c, h_w, w = h.shape

        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2)
        v = self.v(h).flatten(2).transpose(1, 2)

        attn = torch.bmm(q, k) * (c**-0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2)
        out = out.reshape(b, c, h_w, w)

        return residual + self.proj_out(out)


# =====================================================================
# --- 3. Down / Up 巨集區塊 ---
# =====================================================================
class DownBlock2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_downsample: bool,
        add_attention: bool,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ImprovedResnetBlock2D(in_channels, out_channels, temb_channels),
                ImprovedResnetBlock2D(out_channels, out_channels, temb_channels),
            ]
        )
        self.attentions = nn.ModuleList(
            [
                AttentionBlock2D(out_channels) if add_attention else nn.Identity()
                for _ in range(2)
            ]
        )
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.downsample = None

    def forward(self, hidden_states, temb):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            if not isinstance(attn, nn.Identity):
                hidden_states = attn(hidden_states)
            output_states.append(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, tuple(output_states)


class UpBlock2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: tuple[int, int, int],
        temb_channels: int,
        add_upsample: bool,
        add_attention: bool,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ImprovedResnetBlock2D(
                    in_channels + skip_channels[0], out_channels, temb_channels
                ),
                ImprovedResnetBlock2D(
                    out_channels + skip_channels[1], out_channels, temb_channels
                ),
                ImprovedResnetBlock2D(
                    out_channels + skip_channels[2], out_channels, temb_channels
                ),
            ]
        )
        self.attentions = nn.ModuleList(
            [
                AttentionBlock2D(out_channels) if add_attention else nn.Identity()
                for _ in range(3)
            ]
        )
        if add_upsample:
            self.upsample = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.upsample = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb):
        res_hidden_states_list = list(res_hidden_states_tuple)
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_list.pop()

            if hidden_states.shape[-2:] != res_hidden_states.shape[-2:]:
                hidden_states = F.interpolate(
                    hidden_states,
                    size=res_hidden_states.shape[-2:],
                    mode="nearest",
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if not isinstance(attn, nn.Identity):
                hidden_states = attn(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


# =====================================================================
# --- 4. 最佳化通道與層數的 UNet 主類別 ---
# =====================================================================
class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        # 【改進】：擴大基礎通道，縮減為 3 層以保留 Latent 4x4 的豐富語意
        block_out_channels: tuple = (128, 256, 512),
        down_block_types: tuple = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: tuple = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(
            block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            add_attention = "Attn" in block_type

            self.down_blocks.append(
                DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=True if i < len(down_block_types) - 1 else False,
                    add_attention=add_attention,
                )
            )

        mid_ch = block_out_channels[-1]
        self.mid_res1 = ImprovedResnetBlock2D(mid_ch, mid_ch, time_embed_dim)
        self.mid_attn = AttentionBlock2D(mid_ch)
        self.mid_res2 = ImprovedResnetBlock2D(mid_ch, mid_ch, time_embed_dim)

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        # 建立 down path channel list，後續用來正確計算每個 up block 的 skip channels
        down_state_channels = [block_out_channels[0]]
        for i, out_ch in enumerate(block_out_channels):
            down_state_channels += [out_ch, out_ch]
            if i < len(block_out_channels) - 1:
                down_state_channels.append(out_ch)

        skip_channel_groups = []
        tmp_channels = down_state_channels.copy()
        for _ in range(len(block_out_channels)):
            group = tmp_channels[-3:]
            tmp_channels = tmp_channels[:-3]
            skip_channel_groups.append(tuple(group[::-1]))

        for i, block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            add_attention = "Attn" in block_type

            self.up_blocks.append(
                UpBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    skip_channels=skip_channel_groups[i],
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    add_attention=add_attention,
                )
            )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_out_channels[0], eps=1e-5
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        timesteps = timestep * torch.ones(
            sample.shape[0], dtype=timestep.dtype, device=timestep.device
        )

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += tuple(res_samples)

        sample = self.mid_res1(sample, emb)
        sample = self.mid_attn(sample)
        sample = self.mid_res2(sample, emb)

        for upsample_block in self.up_blocks:
            # 配合 3 層架構，彈性調整 pop 數量
            res_samples = down_block_res_samples[-3:]
            down_block_res_samples = down_block_res_samples[:-3]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


# =====================================================================
# --- 5. 數據增強增益版 Dataset ---
# =====================================================================
class ImageDataset(Dataset):

    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = sorted(
            [p for p in self.image_dir.glob("*.png") if p.is_file()]
        )
        # 【改進】：加入 RandomHorizontalFlip 進行基礎數據增強，大幅緩解過擬合
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image


@torch.no_grad()
def generate_and_save_images(unet, vae, scheduler, epoch, device, save_folder):
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

        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(
            alpha_bar_t
        )

        # Dynamic thresholding per image to suppress extreme latent values
        abs_pred_x0 = pred_x0.abs().flatten(1)
        s = torch.quantile(abs_pred_x0, 0.995, dim=1).view(-1, 1, 1, 1)
        s = torch.clamp(s, min=1.0, max=2.0)
        pred_x0 = torch.clamp(pred_x0, -s, s) / s

        direction_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_xt

    # Reverse normalization before decoding into pixel space
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, unbiased=False, keepdim=True)
    x_std = torch.clamp(x_std, min=1e-6)
    x = x * x_std + x_mean

    decoded = vae.decode(x / vae.config.scaling_factor).sample
    decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
    for i in range(decoded.shape[0]):
        out_path = os.path.join(save_folder, f"sample_{epoch}_{i}.png")
        save_image(decoded[i], out_path)

    unet.train()


# =====================================================================
# --- 6. 具備動態裁剪與安全解碼的大批量生成函數 ---
# =====================================================================
@torch.no_grad()
def generate_final_results(
    unet, vae, scheduler, device, results_folder, total_images=3000, batch_size=32
):
    """使用包含動態邊界校正的 DDIM 加速批量生成"""
    unet.eval()
    vae.eval()
    os.makedirs(results_folder, exist_ok=True)
    print(f"\n正在開始生成 {total_images} 張圖片至 {results_folder} 資料夾...")

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
    num_batches = math.ceil(total_images / batch_size)

    for b in tqdm(range(num_batches), desc="Generating Images"):
        current_batch_size = min(batch_size, total_images - generated_count)

        x = torch.randn(current_batch_size, 4, 32, 32, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full(
                (current_batch_size,), t, device=device, dtype=torch.long
            )
            pred_noise = unet(x, t_tensor)

            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_prev = get_alpha_prev(i, timesteps)

            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(
                alpha_bar_t
            )

            # Dynamic thresholding per image to suppress extreme latent values
            abs_pred_x0 = pred_x0.abs().flatten(1)
            s = torch.quantile(abs_pred_x0, 0.995, dim=1).view(-1, 1, 1, 1)
            s = torch.clamp(s, min=1.0, max=2.0)
            pred_x0 = torch.clamp(pred_x0, -s, s) / s

            direction_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_xt

        # Reverse normalization before decoding each generated batch
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, unbiased=False, keepdim=True)
        x_std = torch.clamp(x_std, min=1e-6)
        x = x * x_std + x_mean

        # 分批送進 VAE，防止顯存溢出
        decoded_list = []
        vae_sub_batch = 8
        for sub_idx in range(0, x.shape[0], vae_sub_batch):
            sub_x = x[sub_idx : sub_idx + vae_sub_batch]
            sub_decoded = vae.decode(sub_x / vae.config.scaling_factor).sample
            decoded_list.append(sub_decoded)

        decoded = torch.cat(decoded_list, dim=0)
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0

        for i in range(decoded.shape[0]):
            out_path = os.path.join(results_folder, f"{generated_count:04d}.png")
            save_image(decoded[i], out_path)
            generated_count += 1

    print(f"成功完成！所有圖片已儲存至 {results_folder}")


def train():
    # ========= Hyperparameters ==========
    train_epochs = 100
    batch_size = 16
    gradient_accumulation_steps = 1
    lr = 1e-4
    eval_freq = 1000
    image_dir = "input/ref"
    output_dir = "outputs"
    results_dir = "input/res"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA is not available.")

    save_folder = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_folder, exist_ok=True)

    # ========== Load Pretrained Model ==========
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    # ========== Init ==========
    unet = UNet().to(device)
    unet.train()
    optimizer = torch.optim.Adam(list(unet.parameters()), lr=lr)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )

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
                # 1) VAE latent encoding
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2) Batch-wise latent normalization along channel dim
                latent_mean = latents.mean(dim=1, keepdim=True)
                latent_std = latents.std(dim=1, unbiased=False, keepdim=True)
                latent_std = torch.clamp(latent_std, min=1e-6)
                latents = (latents - latent_mean) / latent_std

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
            elif hasattr(pred, "sample"):
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
            generate_and_save_images(
                unet, vae, noise_scheduler, step, device, save_folder
            )
            unet.train()

    torch.save(unet.state_dict(), os.path.join(save_folder, "unet.pt"))

    # ========== 訓練完成：呼叫 3000 張生成任務 ==========
    generate_final_results(
        unet=unet,
        vae=vae,
        scheduler=noise_scheduler,
        device=device,
        results_folder=results_dir,
        total_images=3000,
        batch_size=32,  # 優化後的網路在 32 運作相當安全流暢
    )


if __name__ == "__main__":
    from huggingface_hub import login
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    login(token=token)
    train()
