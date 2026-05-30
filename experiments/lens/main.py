import os
import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from nnsight import NNsight
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH,
    #dispatch=True,
    device_map=DEVICE
)

# define the prompts
prompt = "The sound of a hammer hitting a wooden surface."
negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator(DEVICE).manual_seed(0)

audios = []
#for i in range(model.text_encoder.config.num_hidden_layers):
#    print(i)
#    print(model.text_encoder.encoder.block[i])
#exit()

# Numerical stability, for no -inf or inf
# model.text_encoder = model.text_encoder.to(torch.float32)

"""
audio = model.generate(
            prompt,
            num_inference_steps=20, # 200
            audio_end_in_s=2.0,
            seed=0
        )

output = audio[0].T.float().cpu().numpy()
print("max audio value:", audio.max().item())
print("max output value:", output.max().item())
sf.write(f"layers/all.wav", output, model.vae.sampling_rate)
exit()
"""

for layer in tqdm(range(model.text_encoder.config.num_hidden_layers)):
    with torch.no_grad():
        with model.generate(
            prompt,
            num_inference_steps=200, # 200
            audio_end_in_s=4.0,
            seed=0
        ):
            # layer = -1
            print(f"layer: {layer}")
            # replace the final_layer_norm input with the text_encoder's output for the layer.
            hidden_state = model.text_encoder.encoder.block[layer].output[0]#.save()
            print("range of hidden state:", hidden_state.min().item(), hidden_state.max().item())
            model.text_encoder.encoder.final_layer_norm.input = hidden_state# [0][:] = hidden_state
            # Save the generated audio
            audio = model.output.audios[0].save()
            # audios.append(audio)
            audio = audio[0].T.float().cpu().numpy()
            sf.write(f"experiments/lens/layers/layer{layer}.wav", audio, model.vae.sampling_rate)