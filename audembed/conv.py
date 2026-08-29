import torch
from torch import nn
from __future__ import annotations

class ConvLayer():
    def __init__(self, module: nn.Module=None, weight=None, bias=None, control=False):
        self.weight = None
        self.bias = None
        if module is not None:
            module = nn.utils.remove_weight_norm(module)
            self.weight = module.weight.detach().clone().cpu()
            self.bias = module.bias.detach().clone().cpu()
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias
        if control:
            self.weight = nn.init.kaiming_uniform_(self.weight)
            self.bias = torch.zeros_like(self.bias)
    
    def __call__(self, x: torch.Tensor):
        return nn.functional.conv1d(x, self.weight, self.bias, padding=self.weight.shape[-1] - 1)

    @staticmethod
    def _combine_conv1d(w1, b1, w2, b2):
        """
        Element-wise (true) convolution.
        Refer to https://stackoverflow.com/a/58357816/
        """
        weight = nn.functional.conv1d(
            w1.permute(1, 0, 2), 
            w2.flip(-1), # flip because this is actually correlation
            padding=w2.shape[-1] - 1
        ).permute(1, 0, 2)
        bias = w2.sum(-1) @ b1 + b2
        return weight, bias

    def __mul__(self, other: ConvLayer | SnakeLinearized):
        if type(other) is ConvLayer:
            weight, bias = self._combine_conv1d(
                self.weight, self.bias, 
                other.weight, other.bias
            )
            return ConvLayer(weight=weight, bias=bias)
        elif type(other) is SnakeLinearized:
            weight, bias = self._combine_conv1d(
                self.weight, self.bias,
                torch.diag(other.weight.detach()).unsqueeze(0), other.bias.detach()
            )
            return weight, bias
        else:
            raise TypeError(f"cannot multiply type ConvLayer by type {type(other)}")
    
    def __imul__(self, other: ConvLayer | SnakeLinearized):
        if type(other) is ConvLayer:
            self.weight, self.bias = self._combine_conv1d(
                self.weight, self.bias, 
                other.weight, other.bias
            )
        elif type(other) is SnakeLinearized:
            self.weight, self.bias = self._combine_conv1d(
                self.weight, self.bias,
                torch.diag(other.weight.detach()).unsqueeze(0), other.bias.detach()
            )
        else:
            raise TypeError(f"cannot multiply type ConvLayer by type {type(other)}")

    def __add__(self, other: ConvLayer):
        return ConvLayer(
            weight = self.weight + other.weight,
            bias = self.bias + other.bias
        )

    def __iadd__(self, other: ConvLayer):
        self.weight += other.weight
        self.bias += other.bias
    
    def __str__(self):
        return f"ConvLayer(weight.shape={self.weight.shape}, bias.shape={self.bias.shape})"

def create_virtual_kernel(convs: list[nn.Module] | tuple[nn.Module], control=False):
    combined = ConvLayer(module=convs[0], control=control)
    for conv in convs[1:]:
        conv_kernel = ConvLayer(module=conv, control=control)
        combined *= conv_kernel
    return combined

class SnakeLinearized(nn.Module):
    def __init__(self, features, module=None, alpha=None, beta=None):
        super().__init__()
        self.features = features
        if module is not None:
            self.alpha = module.alpha.detach().clone()
            self.beta = module.beta.detach().clone()
        else:
            self.alpha = alpha
            self.beta = beta
        self.weight = torch.nn.Parameter(torch.zeros(features))
        self.bias = torch.nn.Parameter(torch.zeros(features))

    def original(self, x):
        return torch.sin(self.alpha * x) ** 2 / self.beta

    def forward(self, x):
        return x * self.weight + self.bias
    

if __name__ == "__main__":
    import data
    x = torch.randn((2, 64))

    layers = [
        ConvLayer(weight=torch.randn((128, 2, 7)), bias=torch.randn((128,))),
        ConvLayer(weight=torch.randn((128, 128, 7)), bias=torch.randn((128,))),
        ConvLayer(weight=torch.randn((128, 128, 7)), bias=torch.randn((128,))),
        ConvLayer(weight=torch.randn((128, 2, 7)), bias=torch.randn((128,))),
        ConvLayer(weight=torch.randn((128, 128, 13)), bias=torch.randn((128,)))
    ]
    combined = layers[0] * layers[1] * layers[2] + layers[3] * layers[4]
    y = layers[2](layers[1](layers[0](x))) + layers[4](layers[3](x))
    y_pred = combined(x)
    data.plot_heatmap_2d(
        y - y_pred, 
        xlabel="frames", 
        ylabel="channels", 
        title="difference in output between original and combined operation"
    )