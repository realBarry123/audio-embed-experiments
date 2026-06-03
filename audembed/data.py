import matplotlib.pyplot as plt
import torch

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

def plot_states(x, title=None):
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

    if x.shape[1] == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(
            x[:, 0].cpu().numpy(),
            x[:, 1].cpu().numpy(),
            c=timesteps.cpu().numpy(),
            cmap="viridis",
            s=25,
            edgecolors="none"
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    elif x.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter(
            x[:, 0].cpu().numpy(),
            x[:, 1].cpu().numpy(),
            x[:, 2].cpu().numpy(),
            c=timesteps.cpu().numpy(),
            cmap="viridis",
            s=25,
            edgecolors="none"
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
    
    else:
        raise ValueError(f"dimensions must be 2 or 3")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("timestep")

    if title is not None:
        ax.set_title(title)

    plt.show()

    return ax

if __name__ == "__main__":
    plot_states(
        pca_reduce(torch.rand((67, 1024)), 3),
        title="Test"
    )