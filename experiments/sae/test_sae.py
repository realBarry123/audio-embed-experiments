import os
import torch
from dotenv import load_dotenv
import soundfile as sf

from audembed import audio_datasets, models

DATASET_NAME = "audioset"

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

if DATASET_NAME == "audioset":
    dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0)
else:
    dataset = audio_datasets.MIRDataset(DATASET_NAME, chunk_duration=1.0)
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.2, 
    batch_size=1,
)

for x in valid_loader:
    audio = x
    break

sf.write(f"experiments/sae/results/{DATASET_NAME}_sample.wav", audio[0].T.cpu(), 44100)

vae = models.VAEWrapper(CACHE_PATH, DEVICE)

vae_latent = vae.encode(audio.to(DEVICE))

del audio
torch.cuda.empty_cache()

with torch.no_grad():
    sae_latent = sae.encode(vae_latent)[0].T # (frame, feature_dim=2048)

torch.save(sae_latent, f"experiments/sae/results/{DATASET_NAME}_sae_latent.pt")

with torch.no_grad():
    vae_latent_recon = sae.decode(sae_latent.T.unsqueeze(0)) # (batch=1, frame, latent_dim=64)

audio_recon = vae.decode(vae_latent_recon).cpu()
audio_recon = audio_recon / audio_recon.abs().max()
sf.write(f"experiments/sae/results/{DATASET_NAME}_recon.wav", audio_recon[0].T.detach().cpu(), 44100)

del audio_recon
torch.cuda.empty_cache()

audio_recon_from_vae = vae.decode(vae_latent).cpu()
audio_recon_from_vae = audio_recon_from_vae / audio_recon_from_vae.abs().max()
sf.write(f"experiments/sae/results/{DATASET_NAME}_recon_from_vae.wav", audio_recon_from_vae[0].T.detach().cpu(), 44100)