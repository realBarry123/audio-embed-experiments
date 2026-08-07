import os
import torch
import numpy as np
from dotenv import load_dotenv
from einops import rearrange
import soundfile as sf

from audembed import models

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")
CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")
if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

state_dict, configs = torch.load("experiments/sae/models/reordered_sae.pt")
print(configs)
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

vae = models.VAEWrapper(CACHE_PATH, DEVICE)

def decode_features(features):
    GAIN = 0.4
    features = torch.Tensor(features).to(DEVICE)
    features = rearrange(features, "d f -> 1 f d")
    with torch.no_grad():
        latent = sae.decode(features).detach()
    x = vae.decode(latent)
    x = rearrange(x * GAIN, "1 t c -> c t").detach().cpu()
    return x

import tkinter as tk

VERSION = "prototype"

FEATURE_FRAMES = 256 # 256 => 11s
PX_SIZE = 5
CANVAS_W = PX_SIZE * FEATURE_FRAMES
CANVAS_H = PX_SIZE * sae.configs["feature_dim"]

pen_size = 0
def set_pen_size(n):
    global pen_size
    pen_size = n

features = np.zeros([sae.configs["feature_dim"], FEATURE_FRAMES], dtype=np.float32)
# features = np.random.uniform(-1, 1, size=[sae.configs["feature_dim"], FEATURE_FRAMES])
# features = np.tile(np.expand_dims(np.linspace(0, 1, num=FEATURE_FRAMES, dtype=np.float32), 0), [sae.configs["feature_dim"], 1])

def _from_rgb(rgb):
    # Source - https://stackoverflow.com/a/51592104
    return "#%02x%02x%02x" % rgb

def to_colors(features):
    black = np.zeros(list(features.shape) + [3,], dtype=np.float32)
    red = np.copy(black)
    red[..., 0] = 255
    blue = np.copy(black)
    blue[..., 2] = 255
    features = np.expand_dims(features, -1)
    return np.where(features > 0, features * red, features * -blue).astype(np.int16)

def reset_canvas(canvas):
    features.fill(0.0)
    canvas.create_rectangle(
        0, 0,
        CANVAS_W, CANVAS_H,
        fill="black",
        width=0
    )

def decode_drawing():
    sf.write(f"experiments/sae/results/draw.wav", decode_features(features), 44100)
    print("Audio saved!")

def mouse_move(e, d, canvas):
    global features
    col = e.x // PX_SIZE
    row = e.y // PX_SIZE
    y0, y1 = row-pen_size, row+pen_size+1
    x0, x1 = col-pen_size, col+pen_size+1
    features[y0: y1, x0: x1] = np.clip(features[y0: y1, x0: x1] + d, -1, 1)

    colors = to_colors(features[y0: y1, x0: x1])
    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            canvas.create_rectangle(
                (x0 + j) * PX_SIZE, (y0 + i) * PX_SIZE, 
                (x0 + j+1) * PX_SIZE, (y0 + i + 1) * PX_SIZE, 
                fill=_from_rgb(tuple(colors[i, j].tolist())),
                outline=_from_rgb(tuple((colors[i, j] * 0.8).astype(np.int16).tolist())),
                width=1
            )

root = tk.Tk()
root.geometry(f"{CANVAS_W + 100}x{CANVAS_H + 150}")
root.title(f"SparsePainter {VERSION}")

really_reset_confirmation = tk.BooleanVar(value=False)
reset_button_text = tk.StringVar(value="Reset")

tk.Label(
    root, 
    text=f"SparsePainter",
    font=("Arial", 16, "bold")
).pack()
tk.Label(
    root, 
    text=f"{VERSION}",
    font=("Arial", 15)
).pack()

drawing_tools = tk.Frame(root)
drawing_tools.pack()

for i in range(3):
    tk.Button(
        drawing_tools,
        text=f"{i*2+1}x{i*2+1}",
        bg="White", 
        command=lambda i=i: set_pen_size(i)
    ).grid(row=0, column=i)

def check_reset_canvas():
    global canvas
    if really_reset_confirmation.get():
        reset_canvas(canvas)
        really_reset_confirmation.set(False)
        reset_button_text.set("Reset")
    else:
        really_reset_confirmation.set(True)
        reset_button_text.set("Really reset??")
        root.after(2000, reset_reset_button)

def reset_reset_button():
    really_reset_confirmation.set(False)
    reset_button_text.set("Reset")

tk.Button(
    drawing_tools,
    textvariable=reset_button_text,
    fg="red",
    command=check_reset_canvas,
    width=13 # characters
).grid(row=0, column=3)

canvas = tk.Canvas(
    root, 
    width=FEATURE_FRAMES*PX_SIZE, 
    height=sae.configs["feature_dim"]*PX_SIZE
)
canvas.pack()
reset_canvas(canvas)

tk.Button(
    root, 
    text="DECODE!!", 
    font=("Arial", 16, "bold"),
    padx=10, pady=10,
    width=15, 
    bg="White", 
    relief="raised",
    command=decode_drawing
).pack()

canvas.bind('<B1-Motion>', lambda e: mouse_move(e, 0.3, canvas))
canvas.bind('<ButtonPress-1>', lambda e: mouse_move(e, 0.3, canvas))
canvas.bind('<B3-Motion>', lambda e: mouse_move(e, -0.3, canvas))
canvas.bind('<ButtonPress-3>', lambda e: mouse_move(e, -0.3, canvas))
root.mainloop()
