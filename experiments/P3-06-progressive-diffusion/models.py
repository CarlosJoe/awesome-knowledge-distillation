import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, time_embed_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.mlp(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks, time_embed_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, time_embed_dim))
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, num_res_blocks, time_embed_dim):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = (in_ch + skip_ch) if i == 0 else out_ch
            self.blocks.append(ResBlock(ch_in, out_ch, time_embed_dim))

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, t_emb)
        return x


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_multipliers=None,
                 num_res_blocks=2, time_embed_dim=256):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4]

        channels = [base_channels * m for m in channel_multipliers]

        self.time_mlp = TimeMLP(time_embed_dim, time_embed_dim)

        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.down_blocks.append(DownBlock(in_ch, out_ch, num_res_blocks, time_embed_dim))

        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_embed_dim)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_embed_dim)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels))):
            in_ch = channels[i + 1] if i < len(channels) - 1 else channels[-1]
            out_ch = channels[i]
            skip_ch = channels[i]
            self.up_blocks.append(UpBlock(in_ch, out_ch, skip_ch, num_res_blocks, time_embed_dim))

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h = self.input_conv(x)

        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skips.append(skip)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


class DiffusionSampler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, num_steps, device, eta=0.0):
        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x = torch.randn(shape, device=device)
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)
            eps_pred = model(x, t_batch)

            alpha_t = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            ) if eta > 0 else 0.0

            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * eps_pred
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def ddim_step(self, model, x, t, t_prev, eta=0.0):
        t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=torch.long)
        eps_pred = model(x, t_batch)

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)

        x0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        ) if eta > 0 else 0.0

        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * eps_pred
        noise = torch.randn_like(x) if sigma > 0 else 0.0
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise

        return x_prev, x0_pred

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
