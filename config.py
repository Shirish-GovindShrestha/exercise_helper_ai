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


# ----- GRU classifier -----
# Load optimized hyperparameters from JSON
try:
    with open("best_lstm_hyperparams.json", "r") as f:
        gru_params = json.load(f)
    GRU_HIDDEN_DIM = gru_params["hidden_dim"]
    GRU_NUM_LAYERS = gru_params["lstm_num_layers"]
    GRU_BIDIRECTIONAL = gru_params["use_bilstm"]
    GRU_DROPOUT = gru_params["dropout"]
    GRU_BATCH_SIZE = gru_params["batch_size"]
    GRU_LR = gru_params["lr"]
    GRU_WEIGHT_DECAY = gru_params["weight_decay"]
except FileNotFoundError:
    # Fallback values if JSON not found
    GRU_HIDDEN_DIM = 96
    GRU_NUM_LAYERS = 4
    GRU_BIDIRECTIONAL = True
    GRU_DROPOUT = 0.10161986531531957
    GRU_BATCH_SIZE = 32
    GRU_LR = 0.00022948836331343134
    GRU_WEIGHT_DECAY = 1.1889698841945615e-05

GRU_EPOCHS = 100
GRU_PATIENCE = 15
GRU_SCHEDULER_GAMMA = 0.9747512556836427


# ----- compute -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- I/O -----
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
AE_BEST = MODEL_DIR / "autoencoder_best.pth"
GRU_BEST = MODEL_DIR / "lstm_classifier_best.pth"




#----Interference----
EXPECTED_SEQUENCE_LENGTH = 45
STEP = 10  # number of frames to skip when creating overlapping sequences
