import torch
import torch.nn as nn

class ExerciseClassifier(nn.Module):
    def __init__(
        self,
        num_classes=5,
        input_dim=111,
        hidden_dim=128,
        lstm_num_layers=1,
        use_bilstm=False,
        dropout=0.2
    ):
        super().__init__()

        self.use_bilstm = use_bilstm

        # --- GRU for temporal modeling ---
        self.lstm = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=use_bilstm,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        print("Drop out value: ", dropout)

        # --- Classifier head ---
        fc_input_dim = hidden_dim * 2 if use_bilstm else hidden_dim
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        # GRU processing
        lstm_out, h_n = self.lstm(x)

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
