from torch import nn

class SAE(nn.Module):
    def __init__(self, latent_dim, feature_dim, do_relu=True, do_norm=True):
        super().__init__()
        self.configs = {
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "do_relu": do_relu,
            "do_norm": do_norm
        }
        self.encoder_linear = nn.Linear(latent_dim, feature_dim, bias=True)
        self.decode = nn.Linear(feature_dim, latent_dim, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.modules.normalization.RMSNorm([feature_dim,])
    
    def encode(self, x):
        x = self.encoder_linear(x)
        if self.configs["do_relu"]: x = self.relu(x)
        if self.configs["do_norm"]: x = self.norm(x)
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h) # (batch, frame, latent_dim=64)
        return x_hat, h
   
class FeatureProbe(nn.Module):
    def __init__(self, feature_dim, logit_dim, bias):
        super().__init__()
        self.configs = {
            "feature_dim": feature_dim,
            "logit_dim": logit_dim,
            "bias": bias
        }
        self.linear = nn.Linear(feature_dim, logit_dim, bias=bias)
    
    def forward(self, x):
        return self.linear(x)