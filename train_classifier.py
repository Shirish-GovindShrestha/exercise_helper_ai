import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from dataset.dataset_loader import load_split, get_label_map
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from models.early_stopping import EarlyStopping
from models.autoencoder import FrameAutoencoder as UnifiedAutoencoder
from models.lstm import ExerciseClassifier



# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.006
PATIENCE = 15
MODEL_SAVE_PATH = Path("models/classifier_best.pth")
AUTOENCODER_PATH = Path("models/autoencoder_best.pth")

# Model architecture config
INPUT_DIM = 99
LATENT_DIM = 16
HIDDEN_DIM = 128
DROPOUT = 0.4

# Autoencoder options
USE_AUTOENCODER = False  # Set to False to train from scratch without pretrained weights
FREEZE_ENCODER = False  # Set to True to freeze encoder, False to fine-tune (only if USE_PRETRAINED_AUTOENCODER=True)
ENCODER_LR_RATIO = 0.1  # Learning rate multiplier for encoder when fine-tuning


# --- Load Data ---
print("📦 Loading data...")
label_map = get_label_map()
NUM_CLASSES = len(label_map)

X_train, y_train = load_split("train")
X_eval, y_eval = load_split("eval")

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
X_eval_t = torch.from_numpy(X_eval).float()
y_eval_t = torch.from_numpy(y_eval).long()

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)
eval_loader = DataLoader(
    TensorDataset(X_eval_t, y_eval_t),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

print(f"✅ Loaded {len(X_train)} training samples, {len(X_eval)} eval samples")
print(f"📊 Number of classes: {NUM_CLASSES}")

autoencoder = None


# --- Load or Initialize Autoencoder ---
if USE_AUTOENCODER:
    print(f"\n🔧 Loading pretrained autoencoder from {AUTOENCODER_PATH}...")
    checkpoint = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
    
    autoencoder = UnifiedAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, dropout=DROPOUT)
    
    # Try to load state dict flexibly
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"⚠️ Missing keys: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)} (ignored)")
    
    autoencoder.to(DEVICE)
    print("✅ Pretrained autoencoder loaded successfully")


# --- Build Classifier ---
print(f"\n🏗️ Building classifier (freeze_encoder={FREEZE_ENCODER})...")
model = ExerciseClassifier(
    autoencoder,
    NUM_CLASSES,
    hidden_dim=HIDDEN_DIM,
    freeze_encoder=FREEZE_ENCODER,
    use_autoencoder=USE_AUTOENCODER

).to(DEVICE)

# Setup optimizer with different learning rates
param_groups = []

if FREEZE_ENCODER:
    if USE_AUTOENCODER:     # only valid if encoder exists
        print("🔒 Encoder frozen — training only LSTM + classifier")
    else:
        print("ℹ️ No encoder — training LSTM + classifier only")

    # Train only LSTM + classifier
    param_groups.append({'params': model.lstm.parameters(), 'lr': LR})
    param_groups.append({'params': model.fc.parameters(), 'lr': LR})

else:
    # Encoder will be trained only if it exists
    if USE_AUTOENCODER:
        encoder_lr = LR * ENCODER_LR_RATIO
        print(f"🔓 Fine-tuning encoder (LR={encoder_lr:.6f}) + LSTM + classifier (LR={LR:.6f})")

        param_groups.append({'params': model.encoder.parameters(), 'lr': encoder_lr})
    else:
        print(f"🆕 No encoder — training LSTM + classifier (LR={LR:.6f})")

    # Always train LSTM + classifier
    param_groups.append({'params': model.lstm.parameters(), 'lr': LR})
    param_groups.append({'params': model.fc.parameters(), 'lr': LR})

# Build optimizer
optimizer = optim.Adam(param_groups, weight_decay=1e-4)

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)


# --- Training Loop ---
print(f"\n🚀 Training for up to {EPOCHS} epochs with early stopping (patience={PATIENCE})...")
best_metrics = {}

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_loss = 0
    
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # Evaluation phase
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in eval_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
    
    # Early stopping based on F1 score
    early_stopping(f1, model)
    
    if early_stopping.early_stop:
        print("🛑 Early stopping triggered.")
        break
    
    # Save best metrics
    if f1 == early_stopping.best_value:
        best_metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": all_labels,
            "preds": all_preds
        }

# Restore best weights
model.load_state_dict(early_stopping.best_model_state)


# --- Final Evaluation ---
print("\n" + "="*60)
print("🏁 Final Evaluation Results (Best Model):")
print("="*60)
print(f"Accuracy : {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall   : {best_metrics['recall']:.4f}")
print(f"F1-score : {best_metrics['f1']:.4f}")
print("="*60)

# Get class names
if isinstance(label_map, dict):
    class_names = list(label_map.keys())
else:
    class_names = list(label_map)

# Confusion Matrix
cm = confusion_matrix(best_metrics["labels"], best_metrics["preds"])
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(
    best_metrics["labels"],
    best_metrics["preds"],
    target_names=class_names,
    digits=4
))

# Save model
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder_state_dict': model.encoder.state_dict(),
    'num_classes': NUM_CLASSES,
    'label_map': label_map,
    'accuracy': best_metrics["accuracy"],
    'precision': best_metrics["precision"],
    'recall': best_metrics["recall"],
    'f1': best_metrics["f1"],
    'use_pretrained_autoencoder': USE_AUTOENCODER,
    'freeze_encoder': FREEZE_ENCODER,
    'config': {
        'input_dim': INPUT_DIM,
        'latent_dim': LATENT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'encoder_lr_ratio': ENCODER_LR_RATIO
    }
}, MODEL_SAVE_PATH)

print(f"\n✅ Best classifier + metrics saved to {MODEL_SAVE_PATH}")
print(f"📈 Confusion matrix saved to confusion_matrix.png")