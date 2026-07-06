import os
import datetime

import torch
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)

with model.trace("_"):
    weights = model.vae.encoder.conv1.weight.data
    torch.save(weights, f"experiments/readingframe/results/conv1_weight.pt")