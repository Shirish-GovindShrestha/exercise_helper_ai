import torch.nn as nn


# --- Autoencoder Model with Dropout ---
class FrameAutoencoder(nn.Module):
    def __init__(self, input_dim=99, latent_dim=64, hidden_dim =128 ,dropout=0.2):
        super().__init__()

        # Encoder with dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Decoder with dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        """
        Extract latent features for classification
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            z: (batch, seq_len, latent_dim)
        """
        batch, seq_len, dim = x.shape
        x_flat = x.reshape(batch * seq_len, dim)
        z_flat = self.encoder(x_flat)
        z = z_flat.view(batch, seq_len, -1)
        return z    

    def forward(self, x):
        batch, seq_len, dim = x.shape
        x_flat = x.reshape(batch * seq_len, dim)

        z = self.encoder(x_flat)
        x_recon = self.decoder(z)

        x_recon = x_recon.reshape(batch, seq_len, dim)
        return x_recon, z
