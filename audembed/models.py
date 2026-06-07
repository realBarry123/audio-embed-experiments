from torch import nn

class SAE(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.encoder_linear = nn.Linear(latent_dim, feature_dim, bias=True)
        self.decoder_linear = nn.Linear(feature_dim, latent_dim, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.modules.normalization.RMSNorm()
    
    def encode(self, x):
        x = self.encoder_linear(x)
        x = self.relu(x)
        x = self.norm(x)
        return x
    
    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

class FeatureProbe(nn.Module):
    def __init__(self, feature_dim, logit_dim):
        self.linear = nn.Linear(feature_dim, logit_dim, bias=True)
    
    def forward(self, x):
        return self.linear(x)