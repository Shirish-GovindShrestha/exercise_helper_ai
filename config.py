"""Hyper-parameters for both models.
Only INPUT_DIM is shared; every other key is model-specific."""
from pathlib import Path
import torch

# ----- shared -----
INPUT_DIM = 99



# ----- auto-encoder -----
AE_LATENT_DIM = 12
AE_DROPOUT = 0.2
AE_HIDDEN = 128          # encoder/decoder internal FC size
AE_BATCH_SIZE = 64
AE_EPOCHS = 20
AE_LR = 1e-3
AE_PATIENCE = 10
AE_BATCH_SIZE = 64

# ----- LSTM classifier -----
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 1
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT = 0.4
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_LR = 1e-3
LSTM_PATIENCE = 15
LSTM_BATCH_SIZE = 32
ENCODER_LR_RATIO = 0.1

# ----- compute -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AUTOENCODER = False
FREEZE_ENCODER = True


# ----- I/O -----
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
AE_BEST = MODEL_DIR / "autoencoder_best.pth"
LSTM_BEST = MODEL_DIR / "lstm_classifier_best.pth"