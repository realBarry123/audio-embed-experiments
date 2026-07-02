
import os
import torch
from torch import nn
from tqdm import tqdm
import torchaudio.transforms as T
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from einops import rearrange

from audembed import audio_datasets, models, audio

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

state_dict, configs, start_epoch = torch.load("experiments/sae/models/probe.pt")
probe = models.FeatureProbe(**configs).to(DEVICE)
probe.load_state_dict(state_dict)

dataset = audio_datasets.MIRDataset("orchset")
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.2, 
    batch_size=1
)

diffusion_model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)

def encode(x):
    with diffusion_model.trace("_"):
        latent = diffusion_model.vae.encoder(x).save() # [batch, param*channel, frame]

    latent = rearrange(latent, "b (p c) f -> b f c p", c=64, p=2)[..., 1] # (B, F, C=64)
    with torch.no_grad():
        features = sae.encode(latent).detach()
    return features

iterator = tqdm(valid_loader)

total_squared_error = torch.zeros(128).to(DEVICE)
total_y = torch.zeros(128).to(DEVICE)
total_y_sq = torch.zeros(128).to(DEVICE)
count = 0

probe.eval()
with torch.no_grad():
    for x in iterator:
        x = x.to(DEVICE)
        x_feat = encode(x).detach()
        y = audio.to_spectrogram(x, target_frames=x_feat.shape[1], target_bins=128).squeeze(0)
        
        y_hat = probe(x_feat).squeeze(0) # (frames, bins)
        squared_error = (y - y_hat) ** 2

        total_squared_error += squared_error.mean(dim=0)
        total_y += y.mean(dim=0)
        total_y_sq += (y ** 2).mean(dim=0)
        count += 1

mean_y = total_y / count
var_y = total_y_sq / count - mean_y ** 2
mse = total_squared_error / count

r2 = 1 - mse / (var_y + 1e-8)
torch.save(r2.cpu(), "experiments/sae/results/probe_r2.pt")