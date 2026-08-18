import os
import torch
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv

from audembed import conv

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

CONV_LAYERS = [
    vae.encoder.conv1,
    vae.encoder.block[0].res_unit1.conv1,
    vae.encoder.block[0].res_unit1.conv2,
    vae.encoder.block[0].res_unit2.conv1,
    vae.encoder.block[0].res_unit2.conv2,
]
virtual_kernel = conv.create_virtual_kernel(CONV_LAYERS, control=False)
print(virtual_kernel.shape)

START = 0
N_PLOTS = 4
CHANNEL = 1
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 8]
fig, axs = plt.subplots(N_PLOTS)
for i in range(N_PLOTS):
    axs[i].plot(virtual_kernel[START+i, CHANNEL, :])
    axs[i].set_title(f"virtual_weight[{START+i}][{CHANNEL}]")
    axs[i].set_xlabel("kernel")
    axs[i].set_ylabel("weight")
fig.tight_layout()
plt.show()
exit()
torch.save(virtual_kernel, "experiments/kernel/results/virtual_kernel.pt")