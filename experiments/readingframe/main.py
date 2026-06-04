import os
import sys
import datetime

import torch
import soundfile as sf
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from tqdm import tqdm
from einops import rearrange

from audembed import data

if len(sys.argv) != 2:
    raise SystemExit(f"Incorrect number of command line arguments")
AUDIO_PATH = f"experiments/readingframe/inputs/{sys.argv[1]}"

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

# Create folder for results
timestamp = str(datetime.datetime.now().strftime("%y-%m-%dT%H:%M:%S"))

#if not os.path.exists(timestamp):
#    os.makedirs(f"experiments/readingframe/results/{timestamp}")
#else:
#    raise ZeroDivisionError("Not even a zero division, time is moving backwards! Try again")

model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)

audio, fs = sf.read(AUDIO_PATH)
audio = torch.Tensor(audio).T.to(DEVICE).half()
print(f"Loaded audio with shape: {str(audio.shape)}") # [2, samples]

latents = []

for t in tqdm(range(0, audio.shape[1]-2048+1, 441)):
    with model.trace("_"):
        latent = model.vae.encoder(audio[:, t:t+2048].unsqueeze(0)).save() # [batch, param*channel, frame]
    
    latent = rearrange(latent, "1 (p c) 1 -> c p", c=64, p=2)
    latents.append(latent[:, 1].detach())
    
latents = torch.stack(latents)
latents = data.pca_reduce(latents, pcs=3)
torch.save(latents, f"experiments/readingframe/results/latents.pt")


with open("experiments/readingframe/results/report.txt", "a") as f:
    f.write(f"{timestamp}: {AUDIO_PATH}")