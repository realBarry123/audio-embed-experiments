import math
import torch
from torch import nn
from einops import rearrange, repeat
import matplotlib.pyplot as plt

TAU = 2 * torch.tensor(math.pi) # 2pi is for the deranged
EPS = 1e-8

class GaborRegression(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernels = rearrange(kernel, "out in ker -> (out in) ker")
        self.batch = self.kernels.shape[0]
        self.kernel_size = self.kernels.shape[1]

        # Mean is the centre of the kernel
        self.mean = nn.Parameter(torch.full((self.batch, 1), self.kernel_size / 2))

        # Init variance based on distributions defined by abs(kernels)
        self.centres = torch.linspace(0.5, self.kernel_size - 0.5, steps=self.kernel_size).unsqueeze(0)
        # self.centres = repeat(self.centres, "ker -> batch ker", batch=self.batch)
        squared_deviations = torch.abs(self.kernels) * (self.centres - self.mean)**2
        self.var = nn.Parameter(squared_deviations.mean(dim=1, keepdim=True))

        # Init frequency as half the number of inflection points per kernel
        critical_points = torch.zeros(self.batch)
        prev_sign = torch.zeros(self.batch)
        for k in range(self.kernel_size-1):
            cur_sign = torch.sign(torch.diff(self.kernels, dim=1)[:, k])
            sign_diff = torch.abs(cur_sign - prev_sign)
            critical_points += torch.where(sign_diff == 2, 1, 0)
            prev_sign = cur_sign
        self.frequency = nn.Parameter(critical_points / 2 / self.kernel_size).unsqueeze(1)

        # Init phase st the max peak aligns with the peak of the cosine function
        self.phase = nn.Parameter(torch.argmax(self.kernels, dim=1, keepdim=True).to(torch.float32))

        # Init scale st: 
        # scale * (1/sqrt(TAU*var)) = kernel_mag => scale = kernel_mag * TAU * var
        kernel_magnitude = torch.max(torch.abs(self.kernels), dim=1, keepdim=True).values
        self.scale = nn.Parameter(kernel_magnitude * TAU * self.var)

    def gaussian(self, x):
        # Normal distribution
        return 1 / torch.sqrt(TAU*self.var) * torch.exp(-0.5 * (x-self.mean)**2 / (self.var+EPS))

    def forward(self, x):
        # x.shape == [256, ker]
        print(self.frequency.shape, x.shape)
        wave = torch.cos(TAU * self.frequency * x - self.phase)
        return self.scale * self.gaussian(x) * wave

    @property
    def r_squared(self):
        y_pred = self.forward(self.centres)
        y = self.kernels
        ss_res = ((y - y_pred) ** 2).sum(dim=1, keepdim=True)
        ss_tot = ((y - y.mean(dim=1, keepdim=True)) ** 2).sum(dim=1, keepdim=True)
        return 1 - ss_res / ss_tot

    def __str__(self):
            return (
                f"GaborRegression(\n"
                f"\tbatch={self.batch},\n"
                f"\tkernel_size={self.kernel_size},\n"
                f"\tmean: {self.mean.shape},\n"
                f"\tvar: {self.var.shape},\n"
                f"\tfrequency: {self.frequency.shape},\n"
                f"\tphase: {self.frequency.shape},\n"
                f"\tscale: {self.scale.shape},\n"
                f"\tcentres: {self.centres.shape}\n"
                f")"
            )
    
    def plot(self, index: int):
        ker_xs = self.centres[0]
        ker_ys = self.kernels[index]
        gabor_xs = torch.linspace(0, self.kernel_size, steps=self.kernel_size * 10)
        gabor_ys = self.forward(gabor_xs.unsqueeze(0))[index].detach()
        plt.scatter(ker_xs, ker_ys)
        plt.plot(gabor_xs, gabor_ys)
        plt.show()


def get_r_squared():
    pass


if __name__ == "__main__":
    kernel = torch.load("experiments/kernel/results/virtual_kernel.pt")
    regression = GaborRegression(kernel)
    print(regression)
    print(regression.r_squared.shape)
    regression.plot(1)
