import torch
from torch import nn
from einops import rearrange

class SAE(nn.Module):
    def __init__(self, latent_dim, feature_dim, do_relu=True, do_norm=True, wandb_id=None, k=None):
        super().__init__()
        self.configs = {
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "do_relu": do_relu,
            "do_norm": do_norm,
            "wandb_id": wandb_id
        }
        self.encoder_linear = nn.Linear(latent_dim, feature_dim, bias=True)
        self.decode = nn.Linear(feature_dim, latent_dim, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.modules.normalization.RMSNorm([feature_dim,])
        self.k = k if k is not None else latent_dim
    
    def encode(self, x):
        x = self.encoder_linear(x)
        if self.configs["do_relu"]: x = self.relu(x)
        if self.configs["do_norm"]: x = self.norm(x)
        if self.k < self.configs["latent_dim"]: 
            values, indices = torch.topk(x, self.k, dim=-1)
            x = torch.zeros_like(x, device=x.device)
            x[indices] = values
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h) # (batch, frame, latent_dim=64)
        return x_hat, h
   
class FeatureProbe(nn.Module):
    def __init__(self, feature_dim, logit_dim, bias):
        super().__init__()
        self.configs = {
            "feature_dim": feature_dim,
            "logit_dim": logit_dim,
            "bias": bias
        }
        self.linear = nn.Linear(feature_dim, logit_dim, bias=bias)
    
    def forward(self, x):
        return self.linear(x)
    

from nnsight.modeling.diffusion import DiffusionModel

class VAEWrapper(nn.Module):
    def __init__(self, cache_path, device):
        self.model = DiffusionModel(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=torch.float32,
            cache_dir=cache_path,
            device_map=device
        )

    def encode(self, audio):
        with self.model.trace("_"):
            latent = self.model.vae.encode(audio).latent_dist.mean.save()
        return rearrange(latent, "b c f -> b f c")

    def decode(self, latent):
        latent = rearrange(latent, "b f c -> b c f")
        with self.model.trace("_"):
            audio = self.model.vae.decode(latent).sample.save()
        return audio