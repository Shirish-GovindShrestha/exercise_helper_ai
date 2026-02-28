import torch
from models.tcn import ExerciseClassifier
from data_processing.dataset_loader import get_label_map
import config

DEVICE = "cpu"
CONF_THRESHOLD = 0.6 

# --- Load label map ---
label_map = get_label_map()
NUM_CLASSES = len(label_map)
class_names = list(label_map) if not isinstance(label_map, dict) else list(label_map.keys())

# --- Load Classifier ---
model_saved = torch.load(config.TCN_BEST, map_location=DEVICE, weights_only=False)
model_config = model_saved.get('config', {})

model = ExerciseClassifier(
    num_classes=NUM_CLASSES,
    input_dim=model_config.get('input_dim', config.INPUT_DIM),
    num_channels=model_config.get('num_channels', config.TCN_NUM_CHANNELS),
    kernel_size=model_config.get('kernel_size', config.TCN_KERNEL_SIZE),
    dropout=model_config.get('dropout', config.TCN_DROPOUT)
).to(DEVICE)
model.load_state_dict(model_saved['model_state_dict'])
model.eval()

# Dummy input (batch_size, seq_len, input_dim) — no hidden state needed for TCN
dummy_input = torch.randn(1, config.EXPECTED_SEQUENCE_LENGTH, config.INPUT_DIM)

torch.onnx.export(
    model,
    dummy_input,
    "saved_models/exercise_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={},  # Fixed batch_size=1 and fixed seq_len
    opset_version=17,
    verbose=True
)
