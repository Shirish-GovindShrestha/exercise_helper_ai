import torch
import torch.nn as nn

input_dim = 99
latent_dim = 16


# --- Exercise Classifier ---
class ExerciseClassifier(nn.Module):
    def __init__(self, autoencoder, num_classes, hidden_dim=128, freeze_encoder=True, use_autoencoder=True):
        super().__init__()
        self.use_autoencoder = use_autoencoder
        self.freeze_encoder = freeze_encoder
        if use_autoencoder:
            self.encoder = autoencoder
            # Freeze/unfreeze encoder
            for p in self.encoder.parameters():
                p.requires_grad = not freeze_encoder
            lstm_input_dim = latent_dim
        else:
            self.encoder = None
            lstm_input_dim = input_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # Option 1: Use encoder
        if self.use_autoencoder:
            if self.freeze_encoder:
                with torch.no_grad():
                    z_seq = self.encoder.encode(x)
            else:
                z_seq = self.encoder.encode(x)

        # Option 2: Skip encoder and pass raw inputs
        else:
            z_seq = x  # (batch, seq_len, raw_input_dim)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(z_seq)
        
        # Use last hidden state for classification
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        logits = self.fc(last_hidden)
        
        return logits
