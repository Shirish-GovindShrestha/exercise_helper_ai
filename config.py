"""Hyper-parameters for both models.
Only INPUT_DIM is shared; every other key is model-specific."""
from pathlib import Path
import torch
import json

# ----- shared -----
INPUT_FEATURE = "combined"  # "landmarks" or "angles" or "combined"
INPUT_DIM = (
    33*3 if INPUT_FEATURE == "landmarks"
    else 12 if INPUT_FEATURE == "angles"
    else 33*3 + 12
)


# ----- LSTM classifier -----
# Load optimized hyperparameters from JSON
try:
    with open("best_lstm_hyperparams.json", "r") as f:
        lstm_params = json.load(f)
    LSTM_HIDDEN_DIM = lstm_params["hidden_dim"]
    LSTM_NUM_LAYERS = lstm_params["lstm_num_layers"]
    LSTM_BIDIRECTIONAL = lstm_params["use_bilstm"]
    LSTM_DROPOUT = lstm_params["dropout"]
    LSTM_BATCH_SIZE = lstm_params["batch_size"]
    LSTM_LR = lstm_params["lr"]
    LSTM_WEIGHT_DECAY = lstm_params["weight_decay"]
except FileNotFoundError:
    # Fallback values if JSON not found
    LSTM_HIDDEN_DIM = 96
    LSTM_NUM_LAYERS = 4
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.10161986531531957
    LSTM_BATCH_SIZE = 32
    LSTM_LR = 0.00022948836331343134
    LSTM_WEIGHT_DECAY = 1.1889698841945615e-05

LSTM_EPOCHS = 100
LSTM_PATIENCE = 15
LSTM_SCHEDULER_GAMMA = 0.9747512556836427


# ----- compute -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- I/O -----
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
AE_BEST = MODEL_DIR / "autoencoder_best.pth"
LSTM_BEST = MODEL_DIR / "lstm_classifier_best.pth"




#----Interference----
EXPECTED_SEQUENCE_LENGTH = 45
STEP = 10  # number of frames to skip when creating overlapping sequences
