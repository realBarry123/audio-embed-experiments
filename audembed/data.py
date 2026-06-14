import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def pca_reduce(x: torch.Tensor, pcs: int) -> torch.Tensor:
    """Reduce dimension of sequence of states using PCA
    
    Args: 
        x: Tensor-like with shape (t, h)
        pcs: dimensions to reduce to
    
    Returns: 
        State reduced to (t, pcs)
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() != 2:
        raise ValueError(f"plot_states expects input of shape (t, h), got {tuple(x.shape)}")
    if pcs > x.shape[1]:
        raise ValueError("pcs must be smaller than or equal to hidden dimension")
    
    x = x - x.mean(dim=0, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        components = vh.T[:, :pcs]
    except AttributeError:
        _, _, v = torch.svd(x)
        components = v[:, :pcs]

    return x @ components

def plot_latent(x, title=None):
    """Plot state trajectories in 2D using PCA.

    Args:
        states: Tensor-like with shape (t, d).
        cmap: Colormap for the timestep gradient.
        title: Optional plot title.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() != 2:
        raise ValueError(f"plot_states expects input of shape (t, d), got {tuple(x.shape)}")
    
    fig = plt.figure()
    timesteps = torch.arange(x.shape[0], dtype=torch.float32)
    colors = plt.get_cmap("viridis")(Normalize(vmin=timesteps.min().item(), vmax=timesteps.max().item())(timesteps.cpu().numpy()))

    if x.shape[1] == 2:
        ax = fig.add_subplot()
        points = x[:, :2].cpu().numpy()
        segments = np.stack([points[:-1], points[1:]], axis=1)
        line = LineCollection(
            segments,
            colors=colors[:-1],
            linewidths=1,
            alpha=0.8,
            zorder=1,
        )
        ax.add_collection(line)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=timesteps.cpu().numpy(),
            cmap="viridis",
            s=10,
            edgecolors="none",
            zorder=2,
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.autoscale_view()

    elif x.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
        points = x[:, :3].cpu().numpy()
        segments = np.stack([points[:-1], points[1:]], axis=1)
        line = Line3DCollection(
            segments,
            colors=colors[:-1],
            linewidths=1,
            alpha=0.8,
            zorder=1,
        )
        ax.add_collection(line)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=timesteps.cpu().numpy(),
            cmap="viridis",
            s=10,
            edgecolors="none",
            zorder=2,
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
    
    else:
        raise ValueError(f"dimensions must be 2 or 3")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("timestep")

    if title is not None:
        ax.set_title(title)

    plt.show()

    return ax

def plot_heatmap_2d(x, title=None, cmap="viridis"):
    if x.dim() != 2:
        raise ValueError(f"plot_heatmap_2d expects input of shape (t, d), got {tuple(x.shape)}")

    # Transpose so imshow's x-axis corresponds to the 0th dimension of the input
    data = x.cpu().numpy().T  # shape -> (d, t)

    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("dim 0 (t)")
    ax.set_ylabel("dim 1 (d)")
    if title is not None:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")
    plt.show()
    return ax

def plot_weight(x, title=None):
    raise NotImplementedError()

if __name__ == "__main__":
    latents = torch.load("experiments/readingframe/results/a.pt", map_location=torch.device('cpu'))
    plot_latent(latents)