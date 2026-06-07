import os
import datetime

import torch
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv

# if len(sys.argv) != 2:
#    raise SystemExit(f"Incorrect number of command line arguments")

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

# Create folder for results
# timestamp = str(datetime.datetime.now().strftime("%y-%m-%dT%H:%M:%S"))

model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)
with model.trace("_"):
    weights = model.vae.encoder.layers[0].weight.data
    print(weights.shape)
    #for layer in model.vae.encoder.layers:
    #    if isinstance(layer, torch.nn.Conv1d):

    
#torch.save(latents, f"experiments/readingframe/results/latents.pt")


#with open("experiments/readingframe/results/report.txt", "a") as f:
#    f.write(f"{timestamp}: {AUDIO_PATH}")