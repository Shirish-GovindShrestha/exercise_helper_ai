import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """
    A single TCN residual block with two dilated causal convolutions,
    weight normalization, ReLU activations, and dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding (applied to left side)

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 1x1 residual connection if channel sizes differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove trailing padding to maintain causal property."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ExerciseClassifier(nn.Module):
    """
    Temporal Convolutional Network (TCN) for exercise classification
    from MediaPipe pose landmark sequences.

    Architecture:
        - Stack of TemporalBlocks with exponentially increasing dilation
        - Global average pooling over temporal dimension
        - Fully connected classifier head

    Args:
        num_classes:      Number of exercise classes
        input_dim:        Feature dimension per frame (e.g. 111 for landmarks+angles)
        num_channels:     List of channel sizes for each TCN layer
        kernel_size:      Convolution kernel size (default: 3)
        dropout:          Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        num_classes=5,
        input_dim=111,
        num_channels=None,
        kernel_size=3,
        dropout=0.2,
        # These params are accepted but ignored (for backward compat with config)
        hidden_dim=128,
        lstm_num_layers=1,
        use_bilstm=False,
    ):
        super().__init__()

        if num_channels is None:
            # Default: 4 layers, wider channels for richer representations
            num_channels = [64, 64, 128, 128]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size,
                stride=1, dilation=dilation, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

        # Classifier head
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)   — same input format as GRU model
        Returns:
            logits: (batch, num_classes)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # TCN temporal processing
        x = self.network(x)

        # Global average pooling over the temporal axis
        x = x.mean(dim=2)

        # Classification
        logits = self.fc(x)
        return logits
