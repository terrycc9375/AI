import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. Dataset loading and preprocessing
# ==========================================
class BrainrotDataset(Dataset):
    def __init__(self, img_dir, img_size=64, is_train=True):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.*"))
        self.img_paths = [
            p
            for p in self.img_paths
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                        degrees=6,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
            ]
        ) if is_train else transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
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
# 2. DDIM components: time encoding and UNet
# ==========================================
def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class ResBlock(nn.Module):
    """Residual block with timestep embedding injection."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Scale shift norm
        t_params = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(t_params, 2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale) + shift   
        # h = self.norm2(h)
        
        h = self.act(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """Self-attention module for non-local feature correlation."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels // 8, 1)
        self.proj = nn.Conv2d(channels // 8, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.bmm(q, k)
        attn = torch.softmax(attn / math.sqrt(self.channels // 8), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, -1, H, W)
        out = self.proj(out)

        return x + self.gamma * out


class UNet(nn.Module):
    """U-Net architecture with timestep embedding, skip connections, and multi-level attention."""
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        out_channels=3,
        time_emb_dim=256,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # down blocks
        self.downs = nn.ModuleList()
        in_ch = base_channels
        down_channels = [in_ch]

        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = nn.ModuleList(
                [
                    ResBlock(in_ch if i == 0 else out_ch, out_ch, self.time_emb_dim)
                    for i in range(num_res_blocks)
                ]
            )
            downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            down_modules = {"blocks": blocks, "downsample": downsample}
            
            if mult in (2, 4):
                down_modules["attn"] = SelfAttention(out_ch)

            self.downs.append(nn.ModuleDict(down_modules))
            in_ch = out_ch
            down_channels.append(in_ch)

        # middle bottleneck
        self.mid1 = ResBlock(in_ch, in_ch, self.time_emb_dim)
        self.mid_attn = SelfAttention(in_ch)
        self.mid2 = ResBlock(in_ch, in_ch, self.time_emb_dim)

        # up blocks
        self.ups = nn.ModuleList()
        up_mults = list(reversed(channel_mults[:-1])) + [channel_mults[0]]

        for i, mult in enumerate(up_mults):
            out_ch = base_channels * mult
            upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

            skip_ch = down_channels[-(i + 1)]
            blocks = nn.ModuleList(
                [
                    ResBlock(
                        out_ch + skip_ch if j == 0 else out_ch, out_ch, self.time_emb_dim
                    )
                    for j in range(num_res_blocks)
                ]
            )
            up_modules = {"upsample": upsample, "blocks": blocks}
            
            if mult in (2, 4):
                up_modules["attn"] = SelfAttention(out_ch)

            self.ups.append(nn.ModuleDict(up_modules))
            in_ch = out_ch

        self.final_norm = nn.GroupNorm(32, in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, timesteps):
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        t_emb = timestep_embedding(timesteps, self.time_emb_dim // 4) # 256
        t_emb = self.time_mlp(t_emb)

        hs = []
        h = self.init_conv(x)

        # down
        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, t_emb)
            
            if "attn" in down:
                h = down["attn"](h)
                
            hs.append(h)
            h = down["downsample"](h)

        # middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # up
        for up in self.ups:
            h = up["upsample"](h)
            skip = hs.pop()

            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(
                    h, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            h = torch.cat([h, skip], dim=1)
            for block in up["blocks"]:
                h = block(h, t_emb)
                
            if "attn" in up:
                h = up["attn"](h)

        h = self.final_norm(h)
        h = self.final_act(h)
        return self.final_conv(h)


class Generator(UNet):
    """Wrapper to match the downstream pipeline configuration."""
    def __init__(self, img_size=64, c_in=3, c_out=3, time_dim=256):
        super().__init__(
            in_channels=c_in,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            out_channels=c_out,
            time_emb_dim=time_dim,
        )

    def forward(self, x_t, t):
        return super().forward(x_t, t)


# ==========================================
# 2.5 Zero-Convolution insertion mechanism
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
        self.zero_conv_in = ZeroConv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x_t, t, control_signal):
        controlled_input = x_t + self.zero_conv_in(control_signal)
        return self.base_gen(controlled_input, t)


# ==========================================
# 3. DDIM main framework and training flow
# ==========================================
class DDIM(nn.Module):
    def __init__(self, img_size=64, num_timesteps=1000):
        super().__init__()
        self.img_size = img_size
        self.T = num_timesteps

        self.gen = Generator(img_size=img_size)

        # Standard linear schedule configuration for diffusion models
        self.register_buffer("beta", torch.linspace(1e-4, 0.02, self.T))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_hat", torch.cumprod(self.alpha, dim=0))

        # ========== EMA (Exponential Moving Average) Setup ==========
        self.ema_decay = 0.999
        self.gen_shadow_state = {}
        for name, param in self.gen.named_parameters():
            if param.requires_grad:
                self.gen_shadow_state[name] = param.data.clone()
        self.gen_original_state = {}

    def update_ema(self):
        for name, param in self.gen.named_parameters():
            if param.requires_grad and name in self.gen_shadow_state:
                self.gen_shadow_state[name] = (
                    self.ema_decay * self.gen_shadow_state[name]
                    + (1.0 - self.ema_decay) * param.data
                )

    def apply_ema_weights(self):
        self.gen_original_state = {}
        for name, param in self.gen.named_parameters():
            if param.requires_grad and name in self.gen_shadow_state:
                self.gen_original_state[name] = param.data.clone()
                param.data = self.gen_shadow_state[name].clone()

    def restore_gen_weights(self):
        for name, param in self.gen.named_parameters():
            if param.requires_grad and name in self.gen_original_state:
                param.data = self.gen_original_state[name]

    def to_device_ema(self, device):
        for name in self.gen_shadow_state:
            self.gen_shadow_state[name] = self.gen_shadow_state[name].to(device)

    def q_sample(self, x_0, t, noise):
        """Forward diffusion process tracking."""
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise

    def train_diffusion(self, dataloader, epochs=10, lr=2e-4, device="cuda"):
        self.to(device)
        self.to_device_ema(device)
        optimizer = torch.optim.AdamW(self.gen.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = nn.MSELoss()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:

            epoch_task = progress.add_task("[cyan]Overall DDIM training", total=epochs)

            for epoch in range(epochs):
                batch_task = progress.add_task(f"[yellow]Epoch {epoch+1}/{epochs}", total=len(dataloader))

                for x_0 in dataloader:
                    x_0 = x_0.to(device)
                    batch_size = x_0.size(0)

                    # Random sample across the 1000 step continuous space
                    t = torch.randint(0, self.T, (batch_size,), device=device)
                    noise = torch.randn_like(x_0)
                    
                    # Target noisy feature map formation
                    x_t = self.q_sample(x_0, t, noise)

                    optimizer.zero_grad()
                    # Model predicts the exact added noise variant
                    noise_pred = self.gen(x_t, t)
                    
                    loss = criterion(noise_pred, noise)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                    optimizer.step()

                    self.update_ema()

                    progress.update(
                        batch_task,
                        advance=1,
                        description=f"MSE Noise Loss: {loss.item():.4f}",
                    )

                progress.remove_task(batch_task)
                scheduler.step()
                progress.update(epoch_task, advance=1)

    @torch.no_grad()
    def sample_ddim(self, x_t, ddim_steps=50, eta=0.0):
        """Deterministic DDIM sampling strategy to accelerate reverse-diffusion."""
        device = x_t.device
        B = x_t.size(0)
        
        # Build sub-sampled time step list down to zero indices
        times = torch.linspace(self.T - 1, 0, ddim_steps, dtype=torch.long, device=device)
        
        for i in range(ddim_steps):
            t_idx = times[i]
            t_vec = torch.full((B,), t_idx, dtype=torch.long, device=device)
            
            # Extract noise prediction from model state
            noise_pred = self.gen(x_t, t_vec)
            
            alpha_hat_t = self.alpha_hat[t_idx].view(-1, 1, 1, 1)
            
            if i + 1 < ddim_steps:
                t_prev_idx = times[i + 1]
                alpha_hat_prev = self.alpha_hat[t_prev_idx].view(-1, 1, 1, 1)
            else:
                alpha_hat_prev = torch.ones_like(alpha_hat_t) # boundary destination
                
            # Compute predicted clean x_0 trajectory component
            pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_hat_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0) # clamping enhances structural clipping consistency
            
            # Compute direction pointing to x_t
            sigma_t = eta * torch.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat_t)) * torch.sqrt(1 - alpha_hat_t / alpha_hat_prev)
            dir_xt = torch.sqrt(1 - alpha_hat_prev - sigma_t**2) * noise_pred
            
            # Reconstruct latent point mapping state
            x_t = torch.sqrt(alpha_hat_prev) * pred_x0 + dir_xt
            if eta > 0.0 and i + 1 < ddim_steps:
                x_t = x_t + sigma_t * torch.randn_like(x_t)
                
        return x_t

    @torch.no_grad()
    def generate_save_and_evaluate_fid(
        self,
        dataloader,
        num_images=2000,
        batch_size=32,
        output_dir="generated_images/",
        device="cuda",
        ddim_steps=50,
    ):
        """Accelerated evaluation via custom deterministic sampling steps."""
        from torchmetrics.image.fid import FrechetInceptionDistance

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[FID & Evaluation] Initializing InceptionV3 metric engine...")
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        self.to(device)

        print(f"[FID & Evaluation] Shifting context state to EMA weights...")
        self.apply_ema_weights()
        self.gen.eval()

        print(f"[FID & Evaluation] Accelerating trajectory execution to {ddim_steps} sub-sampled steps...")
        num_batches = math.ceil(num_images / batch_size)
        saved_count = 0

        try:
            for b in range(num_batches):
                current_batch_size = min(batch_size, num_images - saved_count)
                if current_batch_size <= 0:
                    break

                # Initialize starting tensor using standard Gaussian normal distribution
                x_t = torch.randn(current_batch_size, 3, self.img_size, self.img_size, device=device)
                
                # Dynamic fast generation using the ddim update steps matrix
                x_0_pred = self.sample_ddim(x_t, ddim_steps=ddim_steps, eta=0.0)

                normalized_imgs = (x_0_pred + 1) / 2
                normalized_imgs = torch.clamp(normalized_imgs, 0, 1)

                for i in range(current_batch_size):
                    saved_count += 1
                    img_tensor = normalized_imgs[i].cpu()
                    img = transforms.ToPILImage()(img_tensor)
                    img.save(os.path.join(output_dir, f"{saved_count:04d}.png"))

                fake_imgs_uint8 = (normalized_imgs * 255).to(torch.uint8)
                fid_metric.update(fake_imgs_uint8, real=False)

            print(f"Successfully processed {saved_count} synthetic instances inside {output_dir}")

            print(f"[FID & Evaluation] Mapping standard training set distributions...")
            for x_0 in dataloader:
                x_0 = x_0.to(device)
                real_imgs = (x_0 + 1) / 2
                real_imgs = torch.clamp(real_imgs, 0, 1)
                real_imgs = (real_imgs * 255).to(torch.uint8)
                fid_metric.update(real_imgs, real=True)

            print(f"[FID & Evaluation] Finalizing metric evaluations...")
            fid_score = fid_metric.compute().item()
        finally:
            self.restore_gen_weights()
            print(f"[FID & Evaluation] Restored original core weights safely")

        return fid_score


# ==========================================
# 4. Main execution pipeline
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using runtime configuration device: {device}")

    # Settings
    IMG_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 200 # DDIM benefits heavily from prolonged training stability maps

    # Dataset loader initialization setup
    train_dir = os.path.join("dataset", "trainset")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        print(f"Tip: Place training set files under {train_dir}")

    dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE, is_train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # Initialize pure DDIM framework pipeline (1000 step vector maps)
    ddim_model = DDIM(img_size=IMG_SIZE, num_timesteps=2000)

    if len(dataset) > 0:
        print(f"Starting standard DDIM optimization structure ({EPOCHS} epochs total)...")
        ddim_model.train_diffusion(dataloader, epochs=EPOCHS, device=device)

        # Save standard weights parameters checkpoint mapping
        torch.save(ddim_model.state_dict(), "ddim_base.pth")
        print("Model checkpoint saved as ddim_base.pth")
        print(f"Parameters: {sum(p.numel() for p in ddim_model.parameters()):,}")
        

        # Fast deterministic sampling (50 steps) for image batch tracking and calculation
        output_folder = "generated_images/"
        fid_result = ddim_model.generate_save_and_evaluate_fid(
            eval_dataloader,
            num_images=2000,
            batch_size=BATCH_SIZE,
            output_dir=output_folder,
            device=device,
            ddim_steps=100,
        )
        print(f"\nTraining session finalized! Calculated DDIM FID Score: {fid_result:.4f}")

        # Extend control segment module mappings
        print("\nIntegrating downstream frozen control structure using Zero-Convolution blocks...")
        controlled_gen = ControlledGenerator(ddim_model.gen)
        controlled_gen.to(device)

        torch.save(controlled_gen.state_dict(), "ddim_with_zeroconv.pth")
        print("Successfully formatted pipeline file: ddim_with_zeroconv.pth")
        print("Base generator parameter tensors are locked. Initialization complete!")