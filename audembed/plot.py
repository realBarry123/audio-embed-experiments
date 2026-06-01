import matplotlib.pyplot as plt
import torch


def plot_states(states, ax=None, cmap='viridis', title=None, show=True):
    """Plot state trajectories in 2D using PCA.

    Args:
        states: Tensor-like with shape (t, h).
        ax: Optional matplotlib Axes to draw on.
        cmap: Colormap for the timestep gradient.
        title: Optional plot title.
        show: Whether to call plt.show() before returning.

    Returns:
        The matplotlib Axes containing the scatter plot.
    """
    x = torch.as_tensor(states, dtype=torch.float32)
    if x.dim() != 2:
        raise ValueError(f"plot_states expects input of shape (t, h), got {tuple(x.shape)}")

    x = x - x.mean(dim=0, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        components = vh.T[:, :2]
    except AttributeError:
        _, _, v = torch.svd(x)
        components = v[:, :2]

    projected = x @ components

    if ax is None:
        fig, ax = plt.subplots()

    timesteps = torch.arange(x.shape[0], dtype=torch.float32)
    scatter = ax.scatter(
        projected[:, 0].cpu().numpy(),
        projected[:, 1].cpu().numpy(),
        c=timesteps.cpu().numpy(),
        cmap=cmap,
        s=25,
        edgecolors='none',
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('timestep')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    return ax
