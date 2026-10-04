import os
import torch
from torch import nn
import nnsight
import wandb
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv
import soundfile as sf

from audembed import snake, audio_datasets

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")
CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")
if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

vae = AutoencoderOobleck.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH
).to(DEVICE)

encoder = nnsight.NNsight(vae.encoder)

MODEL_NAME = "snake1lin"

dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0, device=DEVICE)
train_loader, valid_loader = dataset.get_loaders(
    subset=150,
    valid_split=0.2, 
    batch_size=1
)

vae = AutoencoderOobleck.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH
).to(DEVICE)
encoder = nnsight.NNsight(vae.encoder)

N_SAMPLES = 3
dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0, device=DEVICE)
loader = dataset.get_minimal_loader(subset=N_SAMPLES)

def load_snakelin(name):
    state_dict, features, _ = torch.load(f"experiments/kernel/models/{name}.pt")
    model = snake.SnakeLinearized(features).to(DEVICE)
    model.load_state_dict(state_dict)
    print(torch.min(model.weight).item(), torch.max(model.weight).item())
    print(torch.min(model.bias).item(), torch.max(model.bias).item())
    return model

LAYERS = [
    (encoder.block[0].res_unit1.snake1, load_snakelin("snake1lin"))
]

for i, x in enumerate(loader):
    x = x.to(DEVICE)
    x_hat = vae.decoder(encoder(x)[:, :64, :])
    audios = [x[0].T.cpu(), x_hat[0].T.detach().cpu()]
    for layer, replacement in LAYERS:
        with encoder.trace(x):
            layer.output = replacement(layer.input.save())
            x_hat = vae.decoder(encoder.output.save()[:, :64, :])
            audios.append(x_hat[0].T.detach().cpu())

    audios = torch.cat(audios, dim=0)
    sf.write(f"experiments/kernel/results/linearization/{i}.wav", audios, 44100)