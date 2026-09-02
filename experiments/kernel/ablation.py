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

dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0, device=DEVICE)
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.01,
    batch_size=1,
)

encoder = nnsight.NNsight(vae.encoder)
ABLATED = encoder.block[0].res_unit1.snake1
NUM_SAMPLES = 1

i = 0
for x in valid_loader:
    sf.write(f"experiments/kernel/results/ablation/{i}_original.wav", x[0].T.cpu(), 44100)
    x = x.to(DEVICE)
    x_hat = vae.decoder(encoder(x)[:, :64, :])
    sf.write(f"experiments/kernel/results/ablation/{i}_recon.wav", x_hat[0].T.detach().cpu(), 44100)
    with encoder.trace(x):
        ABLATED.output = ABLATED.input.save()
        x_hat = vae.decoder(encoder.output.save()[:, :64, :])
        sf.write(f"experiments/kernel/results/ablation/{i}_ablated_recon.wav", x_hat[0].T.detach().cpu(), 44100)

    i += 1
    if i == NUM_SAMPLES: break