import os, sys
from dotenv import load_dotenv
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint


if len(sys.argv) != 2:
    raise SystemExit(f"Incorrect number of command line arguments")
prompt = sys.argv[1]

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

if DEVICE == "cuda":
  model_half = True

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-3-medium")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(DEVICE)
if model_half:
  model = model.to(torch.float16)
# Set up text and timing conditioning
conditioning = [{
    "prompt": (
        "A dream-like Synthpop instrumental that would accompany "
        "a dream-sequence in a surrealist movie 120 BPM"
    ),
    "seconds_total": 380
}]

# Generate stereo audio
output = generate_diffusion_cond_inpaint(
    model,
    steps=8,
    cfg_scale=1.0,
    conditioning=conditioning,
    sample_size=sample_size,
    sampler_type="pingpong",
    device=DEVICE
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("experiments/lens/results/output.wav", output, sample_rate)
