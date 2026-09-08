import os
import torch
import nnsight
import soundfile as sf
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv

from audembed import audio_datasets

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

N_SAMPLES = 3
dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0, device=DEVICE)
loader = dataset.get_minimal_loader(n_samples=N_SAMPLES)

ABLATED_LAYERS = [
    encoder.block[0].res_unit1.snake1,
    encoder.block[0].snake1
]

for i, x in enumerate(loader):
    x = x.to(DEVICE)
    x_hat = vae.decoder(encoder(x)[:, :64, :])
    audios = [x[0].T.cpu(), x_hat[0].T.detach().cpu()]
    for layer in ABLATED_LAYERS:
        with encoder.trace(x):
            layer.output = layer.input.save()
            x_hat = vae.decoder(encoder.output.save()[:, :64, :])
            audios.append(x_hat[0].T.detach().cpu())

    audios = torch.cat(audios, dim=0)
    sf.write(f"experiments/kernel/results/ablation/{i}.wav", audios, 44100)
    