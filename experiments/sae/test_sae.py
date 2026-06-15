import os
import torch
from torch import nn
from tqdm import tqdm
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from einops import rearrange
import soundfile as sf

from audembed import datasets, models

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

dataset = datasets.MIRDataset("orchset")
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.2, 
    batch_size=1
)

for x in valid_loader:
    audio = x
    break

sf.write(f"experiments/sae/results/{dataset.dataset_name}_sample.wav", audio[0], 44100)

diffusion_model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)

def encode_fn(x):
    with diffusion_model.trace("_"):
        latent = diffusion_model.vae.encoder(x).save() # [batch, param*channel, frame]

    latent = rearrange(latent, "b (p c) f -> b f c p", c=64, p=2)
    return latent[..., 1] # (B, F, C=64)

vae_latent = encode_fn(audio.to(DEVICE))

sae_latent = sae.encode(vae_latent)[0][0].T # (frame, feature_dim=2048)
print(sae_latent.shape)

torch.save(sae_latent, f"experiments/sae/results/{dataset.dataset_name}_sae_latent.pt")