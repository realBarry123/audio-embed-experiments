import os
import random
import torch
import scipy
from einops import rearrange
from dotenv import load_dotenv

from audembed import audio_datasets, models

if not load_dotenv(): raise SystemExit("No .env file found, please make one in the root directory")
CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")
if not CACHE_PATH or not DEVICE: raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

vae = models.VAEWrapper(CACHE_PATH, DEVICE)

state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0)
train_loader, valid_loader = dataset.get_loaders(valid_split=0.2, batch_size=1)

N_OBSERVATIONS = 256
N_CHANNELS = 128

def find_ordering(observations):
    distance = 1 - torch.corrcoef(observations)
    y = scipy.spatial.distance.squareform(distance) # to condensed form
    Z = scipy.cluster.hierarchy.linkage(y, method='average')
    Z_ordered = scipy.cluster.hierarchy.optimal_leaf_ordering(Z, y)
    return scipy.cluster.hierarchy.leaves_list(Z_ordered)


i = 0
observations = []

for x in valid_loader:
    latent = vae.encode(x)
    frame = random.randint(0, latent.shape[1]-1)
    latent = latent[:, frame:frame+1, :]
    with torch.no_grad():
        features = sae.encode(latent).detach()
    observations.append(latent)
    if i % 16 == 0: print(i)
    i += 1
    if i == N_OBSERVATIONS:
        break
print("=== observations complete ===")

observations = torch.stack(observations, dim=1) # (channel, sample)

top_channels = torch.topk(observations.sum(dim=1), k=N_CHANNELS).indices
ordering = find_ordering(observations[top_channels])

sae.decode.weight.data = sae.decode.weight[:, ordering] # no need to change decoder bias
sae.decode.in_features = N_CHANNELS
sae.configs["feature_dim"] = N_CHANNELS
sae.encoder_linear.weight.data = sae.encoder_linear.weight[ordering, :]
sae.encoder_linear.bias.data = sae.encoder_linear.bias[ordering]
sae.encoder_linear.out_features = N_CHANNELS

sae = sae.to("cpu")
torch.save([sae.state_dict(), sae.configs], "experiments/sae/models/reordered_sae.pt")
