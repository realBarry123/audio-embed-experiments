import os
import torch
from torch import nn
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

def get_conv_kernel(conv: nn.Module):
    conv = nn.utils.remove_weight_norm(conv)
    return conv.weight.detach().clone().cpu()

def combine_conv1d(w1, w2):
    """
    Element-wise (true) convolution.
    Refer to https://stackoverflow.com/a/58357816/
    """
    return nn.functional.conv1d(
        w1.permute(1, 0, 2), 
        w2.flip(-1), # flip because this is actually correlation
        stride=1, 
        padding=w1.shape[-1] - 1
    ).permute(1, 0, 2)

def create_virtual_kernel(convs: list[nn.Module] | tuple[nn.Module], control=False):
    combined = get_conv_kernel(convs[0])
    for conv in convs[1:]:
        if control: # load up the dead salmon
            conv_kernel = nn.init.kaiming_uniform_(get_conv_kernel(conv))
        else: 
            conv_kernel = get_conv_kernel(conv)
        combined = combine_conv1d(combined, conv_kernel)
    return combined

CONV_LAYERS = [
    vae.encoder.conv1,
    vae.encoder.block[0].res_unit1.conv1,
    vae.encoder.block[0].res_unit1.conv2,
]
virtual_kernel = create_virtual_kernel(CONV_LAYERS, control=True)
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
torch.save(virtual_kernel, "experiments/kernel/results/virtual_kernel.pt")