import torch
from torch import nn

class FeatureProbe(nn.Module):
    def __init__(self, feature_dim, logit_dim):
        self.linear = nn.Linear(feature_dim, logit_dim, bias=True)
    
    def forward(self, x):
        return self.linear(x)