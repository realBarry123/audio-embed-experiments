import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from audembed import datasets

class SAE(nn.Module):
    def __init__(self, latent_dim, feature_dim, do_relu=True, do_norm=True):
        super(self).__init__()
        self.configs = {
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "do_relu": do_relu,
            "do_norm": do_norm
        }
        self.encoder_linear = nn.Linear(latent_dim, feature_dim, bias=True)
        self.decoder_linear = nn.Linear(feature_dim, latent_dim, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.modules.normalization.RMSNorm()
    
    def encode(self, x):
        x = self.encoder_linear(x)
        x = self.relu(x)
        x = self.norm(x)
        return x
    
    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    

def train_sae(
        model, 
        loader, 
        optim, 
        lamb, 
        encode_fn=None, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True
    ):
    iterator = tqdm(loader, desc=f"E{epoch} Train") if do_tqdm else loader
    total_loss = 0
    total = 0

    for x in iterator:
        x = x.to(device)
        if encode_fn is not None: 
            x = encode_fn(x)

        model.train()
        x_hat, h = model(x)

        loss = nn.functional.mse_loss(x, x_hat) + lamb * torch.linalg.vector_norm(h, dim=-1).mean()
        total_loss += loss.item()
        total += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss / total

def valid_sae(
        model, 
        loader, 
        lamb, 
        encode_fn=None, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True
    ):
    iterator = tqdm(loader, desc=f"E{epoch} Train") if do_tqdm else loader
    total_loss = 0
    total = 0

    for x in iterator:
        x = x.to(device)
        if encode_fn is not None: 
            x = encode_fn(x)

        model.eval()
        with torch.no_grad():
            x_hat, h = model(x)
            loss = nn.functional.mse_loss(x, x_hat) + lamb * torch.linalg.vector_norm(h, dim=-1).mean()
            total_loss += loss.item()
            total += 1
    
    return total_loss / total

