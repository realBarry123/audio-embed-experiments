import math
import torch
from torch import nn, optim
from einops import rearrange
import matplotlib.pyplot as plt

TAU = 2 * math.pi # What's your point? I do what I want. 
EPS = 1e-8

class GaborRegression(nn.Module):
    def __init__(self, kernel):
        # TODO: make buffers
        super().__init__()
        self.kernels = rearrange(kernel, "out in ker -> (out in) ker")
        self.batch = self.kernels.shape[0]
        self.kernel_size = self.kernels.shape[1]

        # Mean is the centre of the kernel
        self.mean = nn.Parameter(torch.full((self.batch, 1), self.kernel_size / 2))

        # Init variance based on distributions defined by max pooled abs(kernels)
        self.centres = torch.linspace(0.5, self.kernel_size - 0.5, steps=self.kernel_size).unsqueeze(0)
        # self.centres = repeat(self.centres, "ker -> batch ker", batch=self.batch)
        POOL_KERNEL = 3
        envelope = nn.functional.max_pool1d(self.kernels.clone().abs(), kernel_size=POOL_KERNEL, stride=1)
        relative_centres = ((self.centres - self.mean.detach().clone())**2)
        if POOL_KERNEL >= 3:
            squared_deviations = envelope * relative_centres[:, POOL_KERNEL//2:-POOL_KERNEL//2+1]
        elif POOL_KERNEL == 2: 
            squared_deviations = envelope * relative_centres[:, -1]
        elif POOL_KERNEL == 1:
            squared_deviations = envelope * relative_centres
        else: 
            raise ValueError("invalid POOL_KERNEL")
        self.var = nn.Parameter(squared_deviations.mean(dim=1, keepdim=True))

        # Init frequency as half the number of inflection points per kernel
        # TODO: use fft instead
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
        wave = torch.cos(TAU * self.frequency * x - self.phase)
        return self.scale * self.gaussian(x) * wave

    def clamp_params(self):
        self.var = nn.Parameter(self.var.clamp(min=0.1))

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

    def _check_nan(self):
        params = [
            (self.mean, "mean"), 
            (self.var, "var"), 
            (self.frequency, "frequency"), 
            (self.phase, "phase"), 
            (self.scale, "scale"), 
        ]
        print("=== checking nans ===")
        for param, name in params:
            is_nan = torch.isnan(param)
            nan_count = torch.zeros(is_nan.shape)
            nan_count[is_nan] = 1
            nan_count = nan_count.sum().item()
            if nan_count > 0:
                print(name, "has", nan_count, "torch.nan")
                continue
            print(name, "ok")
        print("=====================")

def train_gabor_regression(regression: GaborRegression, optimizer: optim.Optimizer):
    regression.train()
    loss = nn.functional.mse_loss( 
        regression.gaussian(regression.centres.detach()),
        regression.kernels.detach()
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    regression.clamp_params()
    return loss.item()

def get_r_squared():
    pass

def _track_nan(regression, param, optimizer): 
    param_history = []
    for i in range(128):
        train_gabor_regression(regression, optimizer=optimizer)
        regression._check_nan()
        param_history.append(param.detach().clone())

    param_history = torch.stack(param_history)
    is_nan = torch.isnan(regression.var)
    nan_progressions = param_history[:, is_nan].T

    for progression in nan_progressions: 
        plt.plot(progression)
    plt.show()

if __name__ == "__main__":
    kernel = torch.load("experiments/kernel/results/virtual_kernel.pt")
    regression = GaborRegression(kernel.to(torch.float32))
    regression.plot(0, include_envelope=True)
    print(regression.r_squared.mean().item())

    LR = 1.5

    optimizer = torch.optim.Adam(params=regression.parameters(), lr=LR)

    for i in range(1024):
        print(train_gabor_regression(regression, optimizer=optimizer))

    regression.plot(0, include_envelope=True)
    print(regression.r_squared.mean().item()) # TODO: this is negative right now :((
