import os
import torch
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv

if not load_dotenv(): 
    raise SystemExit("No .env file found, please make one in the root directory")
CACHE_PATH = os.getenv("CACHE_PATH")
if not CACHE_PATH: 
    raise ValueError("Missing required environment variables: CACHE_PATH")

vae = AutoencoderOobleck.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="vae",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH
).to("cpu")

""" This also seems to work: 
g = vae.encoder.conv1.weight_g.detach().clone().cpu().squeeze(-1)
v = vae.encoder.conv1.weight_v.detach().clone().cpu()
norm_v = v.norm(dim=-1, keepdim=True)
weight = (g / norm_v) * v
"""

conv1 = torch.nn.utils.remove_weight_norm(vae.encoder.conv1)
weight = conv1.weight.detach().clone().cpu()
torch.save(weight, "experiments/kernel/results/conv1_weight.pt")