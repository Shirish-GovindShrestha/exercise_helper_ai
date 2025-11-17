import torch
import torch.nn as nn

class ExerciseClassifier(nn.Module):
    def __init__(
        self,
        autoencoder=None,
        num_classes=5,
        input_dim=33*3,
        hidden_dim=128,
        freeze_encoder=True,
        use_autoencoder=True,
        use_bilstm=False,
        dropout=0.2
    ):
        super().__init__()

        self.use_autoencoder = use_autoencoder
        self.freeze_encoder = freeze_encoder
        self.use_bilstm = use_bilstm

        # --- Encoder setup ---
        if use_autoencoder and autoencoder is not None:
            self.encoder = autoencoder
            # Freeze/unfreeze encoder
            for p in self.encoder.parameters():
                p.requires_grad = not freeze_encoder
        else:
            self.encoder = None

        # --- LSTM for temporal modeling ---
        self.lstm = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=use_bilstm
        )

        # --- Classifier head ---
        fc_input_dim = hidden_dim * 2 if use_bilstm else hidden_dim
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        # --- Encode if using autoencoder ---
        if self.use_autoencoder and self.encoder is not None:
            if self.freeze_encoder:
                with torch.no_grad():
                    z_seq = self.encoder.encode(x)
            else:
                z_seq = self.encoder.encode(x)
        else:
            z_seq = x  # raw input

            

        # --- LSTM ---
        #lstm_out, (h_n, c_n) = self.lstm(z_seq)

        lstm_out, h_n, = self.lstm(z_seq)

        # --- Last hidden state ---
        if self.use_bilstm:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            last_hidden = torch.cat([forward_last, backward_last], dim=1)
        else:
            last_hidden = h_n[-1]

        # --- Classification ---
        logits = self.fc(last_hidden)
        return logits
