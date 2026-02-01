# test_confusion.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from data_processing.dataset_loader import load_split, get_label_map
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import config
from models.lstm import ExerciseClassifier


DEVICE = config.DEVICE
# --- Load label map ---
label_map = get_label_map()
NUM_CLASSES = len(label_map)
if isinstance(label_map, dict):
    class_names = list(label_map.keys())
else:
    class_names = list(label_map)

# --- Load test data ---
X_test, y_test = load_split("test", mode=config.INPUT_FEATURE)

# Convert to tensors and DataLoader
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).long()
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=config.LSTM_BATCH_SIZE,
    shuffle=False
)

# --- Load classifier ---
print(f"Loading model from {config.LSTM_BEST}...")
model_saved = torch.load(config.LSTM_BEST, weights_only=False, map_location=DEVICE)

model = ExerciseClassifier(
    num_classes=NUM_CLASSES,
    input_dim=model_saved['config']['input_dim'],
    hidden_dim=model_saved['config']['hidden_dim'],
    lstm_num_layers=model_saved['config']['lstm_num_layers'],
    use_bilstm=model_saved['config']['use_bilstm'],
    dropout=model_saved['config']['dropout']
).to(DEVICE)

model.load_state_dict(model_saved['model_state_dict'])
model.eval()
print("✅ Model loaded successfully")

# --- Evaluation ---
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# --- Confusion matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()

# --- Classification report ---
print("\n📊 Classification Report (Test Set):")
print(classification_report(all_labels, all_preds, target_names=class_names))
