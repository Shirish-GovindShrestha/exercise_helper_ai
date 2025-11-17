# test_confusion.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from dataset.dataset_loader import load_split, get_label_map
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MODEL_PATH = Path("models/classifier_best.pth")
AUTOENCODER_PATH = Path("models/autoencoder_best.pth")

# --- Load label map ---
label_map = get_label_map()
NUM_CLASSES = len(label_map)
if isinstance(label_map, dict):
    class_names = list(label_map.keys())
else:
    class_names = list(label_map)

# --- Load test data ---
X_test, y_test = load_split("test")
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).long()
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# --- Define models ---
import torch.nn as nn


class PoseAutoencoder(nn.Module):
    def __init__(self, input_dim=99, seq_len=60, latent_dim=54, hidden_dim=128):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_linear = nn.Linear(hidden_dim * 2, latent_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n_forward = h_n[-2]
        h_n_backward = h_n[-1]
        h_n = torch.cat([h_n_forward, h_n_backward], dim=1)
        return self.encoder_linear(h_n)

    def forward(self, x):
        return self.encode(x)


class ExerciseClassifier(nn.Module):
    def __init__(self, autoencoder, num_classes, hidden_dim=128, freeze_encoder=True):
        super().__init__()
        self.encoder = autoencoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        latent_dim = autoencoder.encoder_linear.out_features
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            z = self.encoder.encode(x)
        return self.classifier(z)


# --- Load autoencoder encoder ---
checkpoint = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
autoencoder = PoseAutoencoder(
    input_dim=checkpoint.get("input_dim", 99),
    seq_len=checkpoint.get("seq_len", 60),
    latent_dim=checkpoint.get("latent_dim", 64)
)
state_dict = checkpoint["model_state_dict"]
encoder_only = {k: v for k, v in state_dict.items() if k.startswith("encoder_")}
autoencoder.load_state_dict(encoder_only, strict=False)
autoencoder.to(DEVICE)
autoencoder.eval()

# --- Load classifier ---
model_ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model = ExerciseClassifier(autoencoder, NUM_CLASSES, freeze_encoder=True).to(DEVICE)
model.load_state_dict(model_ckpt['model_state_dict'])
model.eval()

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
