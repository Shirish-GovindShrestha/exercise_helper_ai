import torch
from models.lstm import ExerciseClassifier
from data_processing.dataset_loader import get_label_map
import config

DEVICE = "cpu"
CONF_THRESHOLD = 0.6 

# --- Load label map ---
label_map = get_label_map()
NUM_CLASSES = len(label_map)
class_names = list(label_map) if not isinstance(label_map, dict) else list(label_map.keys())

# --- Load Classifier ---
model_saved = torch.load(config.GRU_BEST, map_location=DEVICE, weights_only=False)
model = ExerciseClassifier(
    num_classes=NUM_CLASSES,
    input_dim=config.INPUT_DIM,
    hidden_dim=config.GRU_HIDDEN_DIM,
    lstm_num_layers=config.GRU_NUM_LAYERS,
    use_bilstm=config.GRU_BIDIRECTIONAL,
    dropout=config.GRU_DROPOUT
).to(DEVICE)
model.load_state_dict(model_saved['model_state_dict'])
model.eval()

# Dummy input (batch_size, seq_len, input_dim)
dummy_input = torch.randn(1, config.EXPECTED_SEQUENCE_LENGTH, config.INPUT_DIM)

# Create initial hidden state for ONNX export to avoid warning
num_directions = 2 if config.GRU_BIDIRECTIONAL else 1
h0 = torch.zeros(config.GRU_NUM_LAYERS * num_directions, 1, config.GRU_HIDDEN_DIM)

torch.onnx.export(
    model,
    (dummy_input, h0),  # Pass both input and hidden state
    "saved_models/exercise_classifier.onnx",
    input_names=["input", "h0"],
    output_names=["output"],
    dynamic_axes={},  # Fixed batch_size=1 and fixed seq_len to avoid GRU warning
    opset_version=17,
    verbose=True
)
