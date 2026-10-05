import math
import torch
from torch import nn
from einops import rearrange
import matplotlib.pyplot as plt

TAU = 2 * math.pi # What's your point? I do what I want. 
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
        squared_deviations = self.kernels.abs() * (self.centres - self.mean)**2
        self.var = nn.Parameter(squared_deviations.mean(dim=1, keepdim=True))

        # Init frequency as half the number of inflection points per kernel
        critical_points = torch.zeros(self.batch)
        prev_sign = torch.zeros(self.batch)
        for k in range(self.kernel_size-1):
            cur_sign = torch.sign(torch.diff(self.kernels, dim=1)[:, k])
            sign_diff = (cur_sign - prev_sign).abs()
            critical_points += torch.where(sign_diff == 2, 1, 0)
            prev_sign = cur_sign
        self.frequency = nn.Parameter(critical_points / 2 / self.kernel_size).unsqueeze(1)

        # Init phase st the max peak aligns with the peak of the cosine function
        self.phase = nn.Parameter(self.kernels.argmax(dim=1, keepdim=True).to(torch.float32))

        # Init scale st: 
        # scale * (1/sqrt(TAU*var)) = kernel_mag => scale = kernel_mag * TAU * var
        kernel_magnitude = self.kernels.abs().max(dim=1, keepdim=True).values
        self.scale = nn.Parameter(kernel_magnitude * TAU * self.var)

    def gaussian(self, x):
        """Probability density of normal distribution based on parameters of self"""
        return 1/torch.sqrt(TAU*self.var) * torch.exp(-0.5 * (x-self.mean)**2 / (self.var+EPS))

    def forward(self, x):
        # x.shape == [batch=256, ker]
        print(self.frequency.shape, x.shape)
        wave = torch.cos(TAU * self.frequency * x - self.phase)
        return self.scale * self.gaussian(x) * wave

    @property
    def r_squared(self):
        """https://en.wikipedia.org/wiki/Coefficient_of_determination"""
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
    
    def plot(self, index: int, include_envelope=False):

        xs = torch.linspace(0, self.kernel_size, steps=self.kernel_size * 10)
        gabor_ys = self.forward(xs.unsqueeze(0))[index].detach()
        plt.plot(xs, gabor_ys, color="blue")
        
        if include_envelope: 
            envelope_ys = (self.scale * self.gaussian(xs.unsqueeze(0)))[index].detach()
            plt.plot(xs, envelope_ys, color="lightblue")

        ker_xs = self.centres[0]
        ker_ys = self.kernels[index]
        plt.stem(ker_xs, ker_ys, linefmt="black", markerfmt=".", basefmt="black")

        plt.show()


def get_r_squared():
    pass


if __name__ == "__main__":
    kernel = torch.load("experiments/kernel/results/virtual_kernel.pt")
    regression = GaborRegression(kernel)
    print(regression)
    print(regression.r_squared.shape)
    regression.plot(1)
