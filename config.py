"""Hyper-parameters for both models.
Only INPUT_DIM is shared; every other key is model-specific."""
from pathlib import Path
import torch

# ----- shared -----
INPUT_FEATURE = "combined"  # "landmarks" or "angles" or "combined"
INPUT_DIM = (
    33*3 if INPUT_FEATURE == "landmarks"
    else 12 if INPUT_FEATURE == "angles"
    else 33*3 + 12
)
#INPUT_DIM  = 9



# ----- auto-encoder -----
AE_LATENT_DIM = 32
AE_DROPOUT = 0.2
AE_HIDDEN = 128          # encoder/decoder internal FC size
AE_BATCH_SIZE = 64
AE_EPOCHS = 50
AE_LR = 1e-2
AE_PATIENCE = 10

# ----- LSTM classifier -----
LSTM_HIDDEN_DIM = 32
LSTM_NUM_LAYERS = 1
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT = 0.4
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_LR = 1e-4
LSTM_PATIENCE = 15
ENCODER_LR_RATIO = 0.1

# ----- compute -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AUTOENCODER = False
FREEZE_ENCODER = False


# ----- I/O -----
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
AE_BEST = MODEL_DIR / "autoencoder_best.pth"
LSTM_BEST = MODEL_DIR / "lstm_classifier_best.pth"




#----Interference----
EXPECTED_SEQUENCE_LENGTH = 75
STEP = 10  # number of frames to skip when creating overlapping sequences