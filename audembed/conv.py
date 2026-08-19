import torch
from torch import nn

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

    def __mul__(self, other):
        weight, bias = self._combine_conv1d(
            self.weight, self.bias, 
            other.weight, other.bias
        )
        return ConvLayer(weight=weight, bias=bias)
    
    def __imul__(self, other):
        self.weight, self.bias = self._combine_conv1d(
            self.weight, self.bias, 
            other.weight, other.bias
        )

    def __add__(self, other):
        return ConvLayer(
            weight = self.weight + other.weight,
            bias = self.bias + other.bias
        )

    def __iadd__(self, other):
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