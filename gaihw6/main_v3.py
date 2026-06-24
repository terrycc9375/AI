import os
import copy
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers.utils import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The parameter 'pretrained' is deprecated.*")


import glob
import time
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from diffusers import DDIMScheduler
from PIL import Image
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)


def create_ema_model(model):
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_parameter, model_parameter in zip(
        ema_model.parameters(), model.parameters()
    ):
        ema_parameter.lerp_(model_parameter, 1.0 - decay)


def apply_cfg_dropout(context, empty_context, dropout_mask):
    return torch.where(dropout_mask, empty_context.expand_as(context), context)


def compute_min_snr_weights(scheduler, timesteps, gamma=5.0):
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    alpha = alphas_cumprod[timesteps].float()
    epsilon = torch.finfo(alpha.dtype).eps
    snr = alpha / (1.0 - alpha).clamp_min(epsilon)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp_min(epsilon)


def compute_balanced_snr_weights(
    scheduler, timesteps, gamma=5.0, min_unweighted_fraction=0.5
):
    min_snr_weights = compute_min_snr_weights(scheduler, timesteps, gamma)
    return min_unweighted_fraction + (
        1.0 - min_unweighted_fraction
    ) * min_snr_weights


def to_image_range(samples):
    return ((samples.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def save_loss_curve(epoch_losses, output_path="loss_curve.png"):
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(epoch_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean denoising loss")
    plt.title("Training loss by epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class BrainrotDataset(Dataset):
    def __init__(self, img_dir, img_size=64, is_train=True):
        self.img_dir = img_dir
        self.data = pd.read_csv("dataset/train.csv")
        self.img_names = self.data["id"]
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomAffine(
                #     degrees=6,
                #     translate=(0.05, 0.05),
                #     scale=(0.95, 1.05),
                #     interpolation=transforms.InterpolationMode.BILINEAR
                # ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        row = self.data.iloc[idx]
        prompt = f"a {row['animal']} and a {row['object']}"
        return self.transform(image), prompt

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        self.norm = nn.GroupNorm(8, query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        residual = x
        b, c, h, w = x.shape
        normalized_x = self.norm(x)
        q = self.to_q(normalized_x.permute(0, 2, 3, 1).reshape(b, h * w, c))
        k = self.to_k(context)
        v = self.to_v(context)
        q = q.reshape(b, h * w, self.heads, c // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, c)
        attention_output = self.to_out(out).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return residual + attention_output

class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, temb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(temb_dim, out_c))
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h += self.time_mlp(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)
    
class UNet2DConditionModel(nn.Module):
    def __init__(self, in_c=3, out_c=3, model_channels=128, channel_multiplier=(1, 2, 3, 4), context_dim=512):
        super().__init__()
        self.time_embed = TimeEmbedding(model_channels * channel_multiplier[0])
        temb_dim = model_channels * channel_multiplier[0] * 4
        self.conv_in = nn.Conv2d(in_c, model_channels * channel_multiplier[0], 3, padding=1)
        
        self.down1 = ResnetBlock(model_channels * channel_multiplier[0], model_channels * channel_multiplier[0], temb_dim)
        self.attn1 = CrossAttention(model_channels * channel_multiplier[0], context_dim)
        self.down1_pool = nn.Conv2d(model_channels * channel_multiplier[0], model_channels * channel_multiplier[1], 3, stride=2, padding=1)
        
        self.down2 = ResnetBlock(model_channels * channel_multiplier[1], model_channels * channel_multiplier[1], temb_dim)
        self.attn2 = CrossAttention(model_channels * channel_multiplier[1], context_dim)
        self.down2_pool = nn.Conv2d(model_channels * channel_multiplier[1], model_channels * channel_multiplier[2], 3, stride=2, padding=1)
        
        self.down3 = ResnetBlock(model_channels * channel_multiplier[2], model_channels * channel_multiplier[2], temb_dim)
        self.attn3 = CrossAttention(model_channels * channel_multiplier[2], context_dim)
        self.down3_pool = nn.Conv2d(model_channels * channel_multiplier[2], model_channels * channel_multiplier[3], 3, stride=2, padding=1)
        
        self.down4 = ResnetBlock(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], temb_dim)
        self.down4_pool = nn.Conv2d(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], 3, stride=2, padding=1)

        self.mid_block1 = ResnetBlock(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], temb_dim)
        self.mid_attn = CrossAttention(model_channels * channel_multiplier[3], context_dim)
        self.mid_block2 = ResnetBlock(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], temb_dim)
        
        self.up4_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], kernel_size=4, stride=2, padding=1)
        self.up4 = ResnetBlock(model_channels * channel_multiplier[3] * 2, model_channels * channel_multiplier[3], temb_dim)

        self.up3_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[3], model_channels * channel_multiplier[2], kernel_size=4, stride=2, padding=1)
        self.up3 = ResnetBlock(model_channels * channel_multiplier[2] * 2, model_channels * channel_multiplier[2], temb_dim)
        self.up3_attn = CrossAttention(model_channels * channel_multiplier[2], context_dim)
        
        self.up2_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[2], model_channels * channel_multiplier[1], kernel_size=4, stride=2, padding=1)
        self.up2 = ResnetBlock(model_channels * channel_multiplier[1] * 2, model_channels * channel_multiplier[1], temb_dim)
        self.up2_attn = CrossAttention(model_channels * channel_multiplier[1], context_dim)
        
        self.up1_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[1], model_channels * channel_multiplier[0], kernel_size=4, stride=2, padding=1)
        self.up1 = ResnetBlock(model_channels * channel_multiplier[0] * 2, model_channels * channel_multiplier[0], temb_dim)
        self.up1_attn = CrossAttention(model_channels * channel_multiplier[0], context_dim)
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels * channel_multiplier[0]),
            nn.SiLU(),
            nn.Conv2d(model_channels * channel_multiplier[0], out_c, 3, padding=1)
        )
        
    def forward(self, x, t, context):
        temb = self.time_embed(t)
        x1 = self.conv_in(x)
        
        # --- Down 1 (CrossAttn) ---
        x1_res = self.down1(x1, temb)
        x1_res = self.attn1(x1_res, context)
        x2 = self.down1_pool(x1_res)
        
        # --- Down 2 (CrossAttn) ---
        x2_res = self.down2(x2, temb)
        x2_res = self.attn2(x2_res, context)
        x3 = self.down2_pool(x2_res)
        
        # --- Down 3 (CrossAttn) ---
        x3_res = self.down3(x3, temb)
        x3_res = self.attn3(x3_res, context)
        x4 = self.down3_pool(x3_res)
        
        # --- Down 4 (Down) ---
        x4_res = self.down4(x4, temb)
        x5 = self.down4_pool(x4_res)
        
        # --- Middle (CrossAttn) ---
        x5 = self.mid_block1(x5, temb)
        x5 = self.mid_attn(x5, context)
        x5 = self.mid_block2(x5, temb)
        
        # --- Up 4 (Up) ---
        h = self.up4_unpool(x5)
        h = torch.cat([h, x4_res], dim=1)
        h = self.up4(h, temb)
        
        # --- Up 3 (CrossAttn) ---
        h = self.up3_unpool(h)
        h = torch.cat([h, x3_res], dim=1)
        h = self.up3(h, temb)
        h = self.up3_attn(h, context)
        
        # --- Up 2 (CrossAttn) ---
        h = self.up2_unpool(h)
        h = torch.cat([h, x2_res], dim=1)
        h = self.up2(h, temb)
        h = self.up2_attn(h, context)
        
        # --- Up 1 (CrossAttn) ---
        h = self.up1_unpool(h)
        h = torch.cat([h, x1_res], dim=1)
        h = self.up1(h, temb)
        h = self.up1_attn(h, context)
        
        return self.out(h)

class UNet2DConditionModel2(nn.Module):
    def __init__(self, in_c=3, out_c=3, model_channels=128, channel_multiplier=(1, 2, 3, 4), context_dim=512):
        super().__init__()
        self.time_embed = TimeEmbedding(model_channels * channel_multiplier[0])
        temb_dim = model_channels * channel_multiplier[0] * 4
        self.conv_in = nn.Conv2d(in_c, model_channels * channel_multiplier[0], 3, padding=1)
        
        self.down1 = ResnetBlock(model_channels * channel_multiplier[0], model_channels * channel_multiplier[0], temb_dim)
        self.down1_pool = nn.Conv2d(model_channels * channel_multiplier[0], model_channels * channel_multiplier[1], 3, stride=2, padding=1)
        
        self.down2 = ResnetBlock(model_channels * channel_multiplier[1], model_channels * channel_multiplier[1], temb_dim)
        self.attn2 = CrossAttention(model_channels * channel_multiplier[1], context_dim)
        self.down2_pool = nn.Conv2d(model_channels * channel_multiplier[1], model_channels * channel_multiplier[2], 3, stride=2, padding=1)
        
        self.down3 = ResnetBlock(model_channels * channel_multiplier[2], model_channels * channel_multiplier[2], temb_dim)
        self.attn3 = CrossAttention(model_channels * channel_multiplier[2], context_dim)
        self.down3_pool = nn.Conv2d(model_channels * channel_multiplier[2], model_channels * channel_multiplier[3], 3, stride=2, padding=1)
        
        self.down4 = ResnetBlock(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], temb_dim)
        self.attn4 = CrossAttention(model_channels * channel_multiplier[3], context_dim)
        self.down4_pool = nn.Conv2d(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], 3, stride=2, padding=1)

        self.mid_block1 = ResnetBlock(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], temb_dim)
        self.mid_attn = CrossAttention(model_channels * channel_multiplier[3], context_dim)
        self.mid_block2 = ResnetBlock(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], temb_dim)
        
        self.up4_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[3], model_channels * channel_multiplier[3], kernel_size=4, stride=2, padding=1)
        self.up4 = ResnetBlock(model_channels * channel_multiplier[3] * 2, model_channels * channel_multiplier[3], temb_dim)
        self.up4_attn = CrossAttention(model_channels * channel_multiplier[3], context_dim)

        self.up3_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[3], model_channels * channel_multiplier[2], kernel_size=4, stride=2, padding=1)
        self.up3 = ResnetBlock(model_channels * channel_multiplier[2] * 2, model_channels * channel_multiplier[2], temb_dim)
        self.up3_attn = CrossAttention(model_channels * channel_multiplier[2], context_dim)
        
        self.up2_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[2], model_channels * channel_multiplier[1], kernel_size=4, stride=2, padding=1)
        self.up2 = ResnetBlock(model_channels * channel_multiplier[1] * 2, model_channels * channel_multiplier[1], temb_dim)
        self.up2_attn = CrossAttention(model_channels * channel_multiplier[1], context_dim)
        
        self.up1_unpool = nn.ConvTranspose2d(model_channels * channel_multiplier[1], model_channels * channel_multiplier[0], kernel_size=4, stride=2, padding=1)
        self.up1 = ResnetBlock(model_channels * channel_multiplier[0] * 2, model_channels * channel_multiplier[0], temb_dim)
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels * channel_multiplier[0]),
            nn.SiLU(),
            nn.Conv2d(model_channels * channel_multiplier[0], out_c, 3, padding=1)
        )
        
    def forward(self, x, t, context):
        temb = self.time_embed(t)
        x1 = self.conv_in(x)
        
        x1_res = self.down1(x1, temb)
        x2 = self.down1_pool(x1_res)
        
        x2_res = self.down2(x2, temb)
        x2_res = self.attn2(x2_res, context)
        x3 = self.down2_pool(x2_res)
        
        x3_res = self.down3(x3, temb)
        x3_res = self.attn3(x3_res, context)
        x4 = self.down3_pool(x3_res)
        
        x4_res = self.down4(x4, temb)
        x4_res = self.attn4(x4_res, context)
        x5 = self.down4_pool(x4_res)
        
        x5 = self.mid_block1(x5, temb)
        x5 = self.mid_attn(x5, context)
        x5 = self.mid_block2(x5, temb)
        
        h = self.up4_unpool(x5)
        h = torch.cat([h, x4_res], dim=1)
        h = self.up4(h, temb)
        h = self.up4_attn(h, context)
        
        h = self.up3_unpool(h)
        h = torch.cat([h, x3_res], dim=1)
        h = self.up3(h, temb)
        h = self.up3_attn(h, context) 
        
        h = self.up2_unpool(h)
        h = torch.cat([h, x2_res], dim=1)
        h = self.up2(h, temb)
        h = self.up2_attn(h, context) 
        
        h = self.up1_unpool(h)
        h = torch.cat([h, x1_res], dim=1)
        h = self.up1(h, temb)
        
        return self.out(h)

class Evaluator():
    def __init__(self, device):
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.inception = inception_v3(pretrained=True).to(device)
        self.inception.eval()
        
    @torch.no_grad()
    def get_inception_features(self, images):
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        features = self.inception(images)
        return features.cpu().numpy()
        
    def calculate_fid(self, real_features, gen_features):
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
    @torch.no_grad()
    def calculate_clip_score(self, images, prompts):
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.clip.get_image_features(pixel_values=inputs['pixel_values'])
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        elif hasattr(image_features, "last_hidden_state"):
            image_features = image_features.last_hidden_state[:, 0]
        text_features = self.clip.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).mean().item()

@torch.no_grad()
def sample(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    context,
    uncond_context,
    device: str = "cuda",
    cfg_scale: float = 3.0,
    steps: int = 50,
):
    b = context.shape[0]
    x = torch.randn(b, 3, 64, 64, device=device)
    scheduler.set_timesteps(num_inference_steps=steps, device=device)

    for t in scheduler.timesteps:
        timestep_batch = torch.full((b,), t, device=device, dtype=torch.long)
        
        # Classifier-Free Guidance (CFG) outputs
        noise_pred_cond = unet(x, timestep_batch, context)
        noise_pred_uncond = unet(x, timestep_batch, uncond_context)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        x = scheduler.step(noise_pred, t, x).prev_sample

        # Dynamic Threshold
        # abs_x = torch.abs(x)
        # s = torch.quantile(abs_x.reshape(b, -1), 0.99, dim=1)
        # s = torch.clamp(s, min=1.0)
        # s = s[:, None, None, None]
        # x = torch.clamp(x, min=-s, max=s) / s
        
    return to_image_range(x)

def train_diffusion(
        unet: UNet2DConditionModel,
        ema_unet: UNet2DConditionModel,
        scheduler: DDIMScheduler, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Adam, 
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        tokenizer, 
        text_encoder, 
        epochs, 
        device
    ):
    unet.train()
    epoch_losses = []
    empty_tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        empty_context = text_encoder(**empty_tokens).last_hidden_state

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        global_task = progress.add_task("Total Training Progress", total=epochs)
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            epoch_task = progress.add_task(f"Epoch {epoch+1}/{epochs}", total=len(dataloader))
            for images, prompts in dataloader:
                images = images.to(device)
                tokens = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    context = text_encoder(**tokens).last_hidden_state
                cfg_dropout_mask = torch.rand(
                    context.shape[0], 1, 1, device=context.device
                ) < 0.15
                context = apply_cfg_dropout(
                    context, empty_context, cfg_dropout_mask
                )
                # t = torch.randint(0, pipeline.num_steps, (images.shape[0],), device=device).long()
                noise = torch.randn_like(images)
                # noisy_images = pipeline.add_noise(images, t, noise)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
                noisy_images = scheduler.add_noise(images, noise, t)
                optimizer.zero_grad()
                pred_noise = unet(noisy_images, t, context)
                per_sample_loss = F.mse_loss(
                    pred_noise, noise, reduction="none"
                ).mean(dim=(1, 2, 3))
                loss_weights = compute_balanced_snr_weights(scheduler, t)
                loss = (per_sample_loss * loss_weights).mean()
                epoch_loss += loss.detach().item()
                batch_count += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                update_ema(ema_unet, unet)
                lr_scheduler.step()
                progress.advance(epoch_task)
            if batch_count > 0:
                epoch_losses.append(epoch_loss / batch_count)
            progress.remove_task(epoch_task)
            progress.advance(global_task)

    return epoch_losses
            
def generate_save_and_evaluate_fid(
    unet,
    scheduler,
    dataloader,
    tokenizer,
    text_encoder,
    evaluator,
    device,
    num_images=2000,
    batch_size=32,
    output_dir="generated_images/",
    ddim_steps=100,
    cfg_scale=2.5,
):
    unet.eval()
    empty_tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        empty_context = text_encoder(**empty_tokens).last_hidden_state
    
    real_features_list = []
    gen_features_list = []
    clip_scores = []
    eval_count = 0
    
    EVALUATE = True
    if EVALUATE:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Evaluating", total=(num_images // batch_size) + 1)
            for images, prompts in dataloader:
                if eval_count >= num_images:
                    break
                images = images.to(device)
                tokens = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    context = text_encoder(**tokens).last_hidden_state
                    uncond_context = empty_context.expand_as(context)
                    gen_images = sample(
                        unet,
                        scheduler,
                        context,
                        uncond_context,
                        cfg_scale=cfg_scale,
                        steps=ddim_steps,
                    )
                real_feats = evaluator.get_inception_features((images + 1.0) / 2.0)
                gen_feats = evaluator.get_inception_features(gen_images)
                real_features_list.append(real_feats)
                gen_features_list.append(gen_feats)
                c_score = evaluator.calculate_clip_score(gen_images, prompts)
                clip_scores.append(c_score)
                eval_count += gen_images.shape[0]
                progress.advance(task)

        real_features = np.concatenate(real_features_list, axis=0)[:num_images]
        gen_features = np.concatenate(gen_features_list, axis=0)[:num_images]
        fid = evaluator.calculate_fid(real_features, gen_features)
        mean_clip = np.mean(clip_scores)
        print(f">> Evaluation finished. Calculated FID: {fid:.4f} | Mean CLIP Score: {mean_clip:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    df_gen = pd.read_csv("dataset/generate.csv")
    all_prompts = [f"a {row['animal']} and a {row['object']}" for _, row in df_gen.iterrows()]
    
    if len(all_prompts) < num_images:
        all_prompts = (all_prompts * (math.ceil(num_images / len(all_prompts))))[:num_images]
    else:
        all_prompts = all_prompts[:num_images]
        
    save_count = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Generating", total=num_images // batch_size + 1)
        for i in range(0, num_images, batch_size):
            batch_prompts = all_prompts[i : i + batch_size]
            
            tokens = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                context = text_encoder(**tokens).last_hidden_state
                uncond_context = empty_context.expand_as(context)
                gen_images = sample(
                    unet,
                    scheduler,
                    context,
                    uncond_context,
                    cfg_scale=cfg_scale,
                    steps=ddim_steps,
                )
                
            for j in range(gen_images.shape[0]):
                img_arr = (gen_images[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                save_count += 1
                Image.fromarray(img_arr).save(os.path.join(output_dir, f"{save_count:06d}.png"))
            
            progress.advance(task)
            
    print(f">> Successfully exported {save_count} images to {output_dir}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("GPU is not detected.")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    IMG_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 200

    train_dir = os.path.join("dataset", "trainset")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        print(f"Tip: Place training set files under {train_dir}")

    dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE, is_train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    unet = UNet2DConditionModel2().to(device)
    ema_unet = create_ema_model(unet)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    total_steps = EPOCHS * len(dataloader)
    warmup_steps = int(total_steps * 0.05)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()
    
    evaluator = Evaluator(device)

    if len(dataset) > 0:
        print(f"Starting standard DDIM optimization structure ({EPOCHS} epochs total)...")
        epoch_losses = train_diffusion(unet, ema_unet, scheduler, dataloader, optimizer, lr_scheduler, tokenizer, text_encoder, EPOCHS, device)
        save_loss_curve(epoch_losses)
        print("Training loss curve saved as loss_curve.png")

        torch.save(unet.state_dict(), "unet.pth")
        print("Model checkpoint saved as unet.pth")
        print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
        
        output_folder = "generated_images/"
        generate_save_and_evaluate_fid(
            ema_unet,
            scheduler,
            eval_dataloader,
            tokenizer,
            text_encoder,
            evaluator,
            device,
            num_images=2000,
            batch_size=BATCH_SIZE,
            output_dir=output_folder,
            ddim_steps=100,
            cfg_scale=2.5,
        )
