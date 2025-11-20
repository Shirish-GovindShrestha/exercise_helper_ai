import torch
from models.lstm import ExerciseClassifier
from models.autoencoder import FrameAutoencoder
from dataset.dataset_loader import load_split, get_label_map
import config

DEVICE = "cpu"
CONF_THRESHOLD = 0.6 

# --- Load label map ---
label_map = get_label_map()
NUM_CLASSES = len(label_map)
class_names = list(label_map) if not isinstance(label_map, dict) else list(label_map.keys())

# --- Load Autoencoder ---
autoencoder = FrameAutoencoder(
    input_dim=config.INPUT_DIM,
    latent_dim=config.AE_LATENT_DIM,
    dropout=config.AE_DROPOUT
).to(DEVICE)
autoencoder.eval()

# --- Load Classifier ---
model_saved = torch.load(config.LSTM_BEST, map_location=DEVICE, weights_only=False)
model = ExerciseClassifier(
    autoencoder=autoencoder,
    num_classes=NUM_CLASSES,
    input_dim=config.INPUT_DIM if not config.USE_AUTOENCODER else config.AE_LATENT_DIM,
    hidden_dim=config.LSTM_HIDDEN_DIM,
    freeze_encoder=config.FREEZE_ENCODER,
    use_autoencoder=config.USE_AUTOENCODER,
    use_bilstm=config.LSTM_BIDIRECTIONAL,
    dropout=config.LSTM_DROPOUT
).to(DEVICE)
model.load_state_dict(model_saved['model_state_dict'])
model.eval()
# Dummy input (seq_len, input_dim)
dummy_input = torch.randn(1, config.EXPECTED_SEQUENCE_LENGTH, config.INPUT_DIM)
torch.onnx.export(
    model,
    dummy_input,
    "exercise_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},  # make batch_size and seq_len dynamic
        "output": {0: "batch_size"}
    },
)
