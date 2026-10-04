import math
import torch
from torch import nn
from einops import rearrange

PI = torch.tensor(math.pi)
EPS = 1e-8

def sign(x: torch.Tensor):
    signs = torch.zeros(x.shape)
    signs = torch.where(x < 0, -1, 0)
    signs = torch.where(x > 0, 1, 0)
    return signs

class GaborRegression(nn.Module):
    def __init__(self, kernel):
        super.__init__()
        self.kernels = rearrange(kernel, "out in ker -> (out in) ker")
        self.batch = self.kernels.shape[0]
        self.kernel_size = self.kernels.shape[1]

        # Mean is the centre of the kernel
        self.mean = nn.Parameter(torch.full((self.batch,), self.kernel_size / 2))

        # Init variance st height of the peak of Gaussian = maximum magnitude of the kernel
        # kernel_magnitude = 1 / sqrt(2 * pi * var) => var = 1 / (2 * pi * kernel_magnitude^2)
        kernel_magnitude = torch.max(torch.abs(self.kernels), dim=1, keepdim=False)
        self.var = nn.Parameter(1 / (2 * PI * (kernel_magnitude ** 2)))

        # Init frequency as the number of zero crossings
        zero_crossings = torch.zeros(self.batch)
        prev_sign = torch.zeros(self.batch)
        for k in range(self.kernels):
            cur_sign = sign(self.kernels[:, k])
            sign_diff = torch.abs(cur_sign - prev_sign)
            zero_crossings += torch.where(sign_diff == 2, 1, 0)
            prev_sign = cur_sign
        self.frequency = nn.Parameter(zero_crossings / 2 / self.kernel_size)

        # Init phase st the max peak aligns with the peak of the cosine function
        self.phase = nn.Parameter(torch.argmax(self.kernels, dim=1))

        # Account for kernel scaling
        self.scale = nn.Parameter(torch.ones(self.batch))

    def gaussian(self, x):
        # Normal distribution
        return 1 / torch.sqrt(2*PI*self.var) * torch.exp(-0.5 * (x-self.mean)**2 / (self.var+EPS))

    def forward(self, x):
        wave = torch.cos(2 * PI * self.frequency * x - self.phase)
        return self.scale * self.gaussian(x) * wave

def get_r_squared():
    pass