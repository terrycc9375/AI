import os
import glob
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
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
from PIL import Image
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

class BrainrotDataset(Dataset):
    def __init__(self, img_dir, img_size=64, is_train=True):
        self.img_dir = img_dir
        self.data = pd.read_csv("dataset/train.csv")
        self.img_names = self.data["id"]
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=6,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
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
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        b, c, h, w = x.shape
        q = self.to_q(x.permute(0, 2, 3, 1).reshape(b, h * w, c))
        k = self.to_k(context)
        v = self.to_v(context)
        q = q.reshape(b, h * w, self.heads, c // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, c)
        return self.to_out(out).reshape(b, h, w, c).permute(0, 3, 1, 2)

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
    def __init__(self, in_c=3, out_c=3, model_channels=128, context_dim=512):
        super().__init__()
        self.time_embed = TimeEmbedding(model_channels)
        temb_dim = model_channels * 4
        self.conv_in = nn.Conv2d(in_c, model_channels, 3, padding=1)
        
        self.down1 = ResnetBlock(model_channels, model_channels, temb_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1)
        
        self.down2 = ResnetBlock(model_channels * 2, model_channels * 2, temb_dim)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 4, 3, stride=2, padding=1)
        
        self.down3 = ResnetBlock(model_channels * 4, model_channels * 4, temb_dim)
        self.attn3 = CrossAttention(model_channels * 4, context_dim)
        self.down3_pool = nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1)
        
        self.mid_block1 = ResnetBlock(model_channels * 4, model_channels * 4, temb_dim)
        self.mid_attn = CrossAttention(model_channels * 4, context_dim)
        self.mid_block2 = ResnetBlock(model_channels * 4, model_channels * 4, temb_dim)
        
        self.up3_unpool = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1)
        self.up3 = ResnetBlock(model_channels * 8, model_channels * 4, temb_dim)
        self.up3_attn = CrossAttention(model_channels * 4, context_dim)
        
        self.up2_unpool = nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResnetBlock(model_channels * 4, model_channels * 2, temb_dim)
        
        self.up1_unpool = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up1 = ResnetBlock(model_channels * 2, model_channels, temb_dim)
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_c, 3, padding=1)
        )
        
    def forward(self, x, t, context):
        temb = self.time_embed(t)
        x1 = self.conv_in(x)
        
        x1_res = self.down1(x1, temb)
        x2 = self.down1_pool(x1_res)
        
        x2_res = self.down2(x2, temb)
        x3 = self.down2_pool(x2_res)
        
        x3_res = self.down3(x3, temb)
        x3_res = self.attn3(x3_res, context)
        x4 = self.down3_pool(x3_res)
        
        x4 = self.mid_block1(x4, temb)
        x4 = self.mid_attn(x4, context)
        x4 = self.mid_block2(x4, temb)
        
        h = self.up3_unpool(x4)
        h = torch.cat([h, x3_res], dim=1)
        h = self.up3(h, temb)
        h = self.up3_attn(h, context)
        
        h = self.up2_unpool(h)
        h = torch.cat([h, x2_res], dim=1)
        h = self.up2(h, temb)
        
        h = self.up1_unpool(h)
        h = torch.cat([h, x1_res], dim=1)
        h = self.up1(h, temb)
        
        return self.out(h)

class DiffusionPipeline(nn.Module):
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        
    @torch.no_grad()
    def sample(self, context, cfg_scale=3.0, steps=50):
        device = next(self.model.parameters()).device
        b = context.shape[0]
        x = torch.randn(b, 3, 64, 64, device=device)
        uncond_context = torch.zeros_like(context)
        
        timesteps = torch.linspace(self.num_steps - 1, 0, steps, dtype=torch.long, device=device)
        for i, t_idx in enumerate(timesteps):
            t = torch.full((b,), t_idx, device=device, dtype=torch.long)
            
            noise_pred_cond = self.model(x, t, context)
            noise_pred_uncond = self.model(x, t, uncond_context)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            alpha_cumprod = self.alphas_cumprod[t_idx]
            alpha_cumprod_prev = self.alphas_cumprod[timesteps[i+1]] if i < steps - 1 else torch.tensor(1.0, device=device)
            
            x_0_pred = (x - torch.sqrt(1.0 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev) * noise_pred
            x = torch.sqrt(alpha_cumprod_prev) * x_0_pred + dir_xt
            
        return (x + 1.0) / 2.0
    
class Evaluator:
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
        text_features = self.clip.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).mean().item()

def train_diffusion(unet, pipeline, dataloader, optimizer, tokenizer, text_encoder, epochs, device):
    unet.train()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        global_task = progress.add_task("Total Training Progress", total=epochs)
        for epoch in range(epochs):
            epoch_task = progress.add_task(f"Epoch {epoch+1}/{epochs}", total=len(dataloader))
            for images, prompts in dataloader:
                images = images.to(device)
                tokens = tokenizer(prompts, padding="max_length", max_length=77, return_tensors="pt")
                with torch.no_grad():
                    context = text_encoder(tokens.input_ids.to(device)).last_hidden_state
                if np.random.rand() < 0.15:
                    context = torch.zeros_like(context)
                t = torch.randint(0, pipeline.num_steps, (images.shape[0],), device=device).long()
                noise = torch.randn_like(images)
                noisy_images = pipeline.add_noise(images, t, noise)
                optimizer.zero_grad()
                pred_noise = unet(noisy_images, t, context)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()
                progress.advance(epoch_task)
            progress.remove_task(epoch_task)
            progress.advance(global_task)
            
def generate_save_and_evaluate_fid(unet, pipeline, dataloader, tokenizer, text_encoder, evaluator, device, num_images=2000, batch_size=32, output_dir="generated_images/", ddim_steps=100):
    unet.eval()
    os.makedirs(output_dir, exist_ok=True)
    real_features_list = []
    gen_features_list = []
    clip_scores = []
    count = 0
    for images, prompts in dataloader:
        if count >= num_images:
            break
        images = images.to(device)
        tokens = tokenizer(prompts, padding="max_length", max_length=77, return_tensors="pt")
        with torch.no_grad():
            context = text_encoder(tokens.input_ids.to(device)).last_hidden_state
        gen_images = pipeline.sample(context, cfg_scale=3.0, steps=ddim_steps)
        for j in range(gen_images.shape[0]):
            if count >= num_images:
                break
            img_arr = (gen_images[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_arr).save(os.path.join(output_dir, f"gen_{count}.png"))
            count += 1
        real_feats = evaluator.get_inception_features((images + 1.0) / 2.0)
        gen_feats = evaluator.get_inception_features(gen_images)
        real_features_list.append(real_feats)
        gen_features_list.append(gen_feats)
        c_score = evaluator.calculate_clip_score(gen_images, prompts)
        clip_scores.append(c_score)
    real_features = np.concatenate(real_features_list, axis=0)[:num_images]
    gen_features = np.concatenate(gen_features_list, axis=0)[:num_images]
    fid = evaluator.calculate_fid(real_features, gen_features)
    mean_clip = np.mean(clip_scores)
    return fid

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("GPU is not detected.")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    IMG_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 1

    train_dir = os.path.join("dataset", "trainset")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        print(f"Tip: Place training set files under {train_dir}")

    dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_dataset = BrainrotDataset(img_dir=train_dir, img_size=IMG_SIZE, is_train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    unet = UNet2DConditionModel().to(device)
    pipeline = DiffusionPipeline(unet, num_steps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()
    
    evaluator = Evaluator(device)

    if len(dataset) > 0:
        print(f"Starting standard DDIM optimization structure ({EPOCHS} epochs total)...")
        train_diffusion(unet, pipeline, dataloader, optimizer, tokenizer, text_encoder, EPOCHS, device)

        torch.save(unet.state_dict(), "ddim_base.pth")
        print("Model checkpoint saved as ddim_base.pth")
        print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
        
        output_folder = "generated_images/"
        fid_result = generate_save_and_evaluate_fid(
            unet,
            pipeline,
            eval_dataloader,
            tokenizer,
            text_encoder,
            evaluator,
            device,
            num_images=2000,
            batch_size=BATCH_SIZE,
            output_dir=output_folder,
            ddim_steps=100,
        )
        print(f"\nTraining session finalized! Calculated DDIM FID Score: {fid_result:.4f}")