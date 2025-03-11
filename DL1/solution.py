import torch
import torch.nn


def encode(x, d, V, W, b, a):
    x = x.reshape(x.shape[0], -1)
    return torch.max(torch.zeros(V.shape[1]), torch.mm(x, V) + b)

def decode(z, d, V, W, b, a):
    return (torch.mm(z, W) + a).view(z.shape[0], d, d)

def get_squared_loss(x, model):
    x_rec = model(x)
    rec = torch.sum((x - x_rec) ** 2)
    return rec / x.shape[0]

def get_squared_loss_with_sparsity(x, model, lam=0.1):
    z = model.encode(x)

    x_rec = model.decode(z)

    rec = torch.sum((x - x_rec) ** 2)
    sp = torch.sum(torch.abs(z))
    return (rec + lam * sp) / x.shape[0]

def sparsity_rate(x, model):
    z = model.encode(x)
    zeros = torch.sum(z != 0)

    return zeros / z.nelement()

def anomaly_score(x, x_rec):
    return ((x_rec - x) ** 2).sum(axis=-1).sum(axis=-1)

class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # any nonlinearity is fine here
        self.swish = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        return x

def get_squared_loss_denoising(x, model, noise=0.1):
    x_noisy = x + noise * torch.randn_like(x)
    x_rec = model(x_noisy)
    rec = torch.sum((x - x_rec) ** 2)
    return rec / x.shape[0]
