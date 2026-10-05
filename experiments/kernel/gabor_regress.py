import math
import torch
from torch import nn
from einops import rearrange, repeat

TAU = 2 * torch.tensor(math.pi) # 2pi is for the deranged
EPS = 1e-8

def sign(x: torch.Tensor):
    signs = torch.zeros(x.shape)
    signs = torch.where(x < 0, -1, 0)
    signs = torch.where(x > 0, 1, 0)
    return signs

class GaborRegression(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernels = rearrange(kernel, "out in ker -> (out in) ker")
        self.batch = self.kernels.shape[0]
        self.kernel_size = self.kernels.shape[1]

        # Mean is the centre of the kernel
        self.mean = nn.Parameter(torch.full((self.batch,), self.kernel_size / 2))

        # Init variance based on distributions defined by abs(kernels)
        centres = torch.linspace(0.5, self.kernel_size - 0.5, steps=self.kernel_size)
        centres = repeat(centres, "ker -> batch ker", batch=self.batch)
        squared_deviations = torch.abs(self.kernels) * (centres - self.mean.unsqueeze(1))**2
        self.var = nn.Parameter(squared_deviations.mean(dim=1))

        # Init frequency as the number of zero crossings
        zero_crossings = torch.zeros(self.batch)
        prev_sign = torch.zeros(self.batch)
        for k in range(self.kernel_size):
            cur_sign = sign(self.kernels[:, k])
            sign_diff = torch.abs(cur_sign - prev_sign)
            zero_crossings += torch.where(sign_diff == 2, 1, 0)
            prev_sign = cur_sign
        self.frequency = nn.Parameter(zero_crossings / 2 / self.kernel_size)

        # Init phase st the max peak aligns with the peak of the cosine function
        self.phase = nn.Parameter(torch.argmax(self.kernels, dim=1).to(torch.float32))

        # Init scale st: 
        # scale * (1/sqrt(TAU*var)) = kernel_mag => scale = kernel_mag * TAU * var
        kernel_magnitude = torch.max(torch.abs(self.kernels), dim=1, keepdim=False).values
        self.scale = nn.Parameter(kernel_magnitude * TAU * self.var)

    def __str__(self):
        return (
            f"GaborRegression(\n"
            f"\tbatch={self.batch},\n"
            f"\tkernel_size={self.kernel_size},\n"
            f"\tmean: {self.mean.shape},\n"
            f"\tvar: {self.var.shape},\n"
            f"\tfrequency: {self.frequency.shape},\n"
            f"\tphase: {self.frequency.shape},\n"
            f"\tscale: {self.scale.shape}\n"
            f")"
        )

    def gaussian(self, x):
        # Normal distribution
        return 1 / torch.sqrt(TAU*self.var) * torch.exp(-0.5 * (x-self.mean)**2 / (self.var+EPS))

    def forward(self, x):
        wave = torch.cos(TAU * self.frequency * x - self.phase)
        return self.scale * self.gaussian(x) * wave

def get_r_squared():
    pass


if __name__ == "__main__":
    kernel = torch.load("experiments/kernel/results/conv1_weight.pt")
    regression = GaborRegression(kernel)
    print(regression)