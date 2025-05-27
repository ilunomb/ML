import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

# Encoder del VAE
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder del VAE
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        return self.sigmoid(self.fc2(h))

# VAE completo
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def loss_function(x, x_hat, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo VAE con una sola barra de progreso para todas las epochs.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {"train_loss": [], "val_loss": []}

        outer_bar = tqdm(range(1, epochs + 1), desc="Entrenando VAE", position=0, leave=True, ncols=120)
        
        for epoch in outer_bar:
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                x_hat, mu, logvar = self(x)
                loss = self.loss_function(x, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_hat, mu, logvar = self(x)
                    loss = self.loss_function(x, x_hat, mu, logvar)
                    val_loss += loss.item()

            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            outer_bar.set_postfix(epoch=epoch, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        return history
