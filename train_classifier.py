import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_processing.dataset_loader import load_split, get_label_map
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from models.early_stopping import EarlyStopping
from models.autoencoder import FrameAutoencoder as UnifiedAutoencoder
from models.lstm import ExerciseClassifier
import config
from sklearn.decomposition import PCA


DEVICE = config.DEVICE


# --- Load Data ---
print("📦 Loading data...")
label_map = get_label_map()
NUM_CLASSES = len(label_map)

X_train, y_train = load_split("train", mode=config.INPUT_FEATURE)
X_eval, y_eval = load_split("eval", mode=config.INPUT_FEATURE)

'''N, T, F = X_train.shape
X_train_2d = X_train.reshape(N*T, F)

# Fit PCA
pca = PCA(n_components=config.INPUT_DIM)   # Use 9 because you found optimal
X_train_pca_2d = pca.fit_transform(X_train_2d)

# Transform eval
M = X_eval.shape[0]
X_eval_pca_2d = pca.transform(X_eval.reshape(M*T, F))


# Reshape back to (N, T, 9)
X_train_pca = X_train_pca_2d.reshape(N, T, config.INPUT_DIM)
X_eval_pca = X_eval_pca_2d.reshape(M, T, config.INPUT_DIM)

import joblib
joblib.dump(pca, "pca_model.joblib")





print("PCA shapes:", X_train_pca.shape, X_eval_pca.shape)'''

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
X_eval_t = torch.from_numpy(X_eval).float()
y_eval_t = torch.from_numpy(y_eval).long()



train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=config.LSTM_BATCH_SIZE,
    shuffle=True
)
eval_loader = DataLoader(
    TensorDataset(X_eval_t, y_eval_t),
    batch_size=config.LSTM_BATCH_SIZE,
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
if config.USE_AUTOENCODER:
    print(f"\n🔧 Loading pretrained autoencoder from {config.AE_BEST}...")
    checkpoint = torch.load(config.AE_BEST, map_location=DEVICE, weights_only=False)
    
    autoencoder = UnifiedAutoencoder(
        input_dim=config.INPUT_DIM,
        latent_dim=config.AE_LATENT_DIM,
        dropout=config.AE_DROPOUT
    )
    
    # Try to load state dict flexibly
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"⚠️ Missing keys: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)} (ignored)")
    
    autoencoder.to(DEVICE)
    print("✅ Pretrained autoencoder loaded successfully")


print(f"\n🏗️ Building classifier (freeze_encoder={config.FREEZE_ENCODER})...")
model = ExerciseClassifier(
    autoencoder=autoencoder,
    num_classes=NUM_CLASSES,
    input_dim=config.INPUT_DIM if not config.USE_AUTOENCODER else config.AE_LATENT_DIM, # raw landmark input dimension
    hidden_dim=config.LSTM_HIDDEN_DIM,
    lstm_num_layers=config.LSTM_NUM_LAYERS,
    freeze_encoder=config.FREEZE_ENCODER,
    use_autoencoder=config.USE_AUTOENCODER,
    use_bilstm=config.LSTM_BIDIRECTIONAL,
    dropout=config.LSTM_DROPOUT
).to(config.DEVICE)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# Setup optimizer with different learning rates
param_groups = []

# Always train LSTM + classifier
param_groups.append({'params': model.lstm.parameters(), 'lr': config.LSTM_LR, 'name': 'LSTM'})
param_groups.append({'params': model.fc.parameters(), 'lr': config.LSTM_LR, 'name': 'FC'})

# Train/fine-tune encoder if needed
if config.USE_AUTOENCODER and not config.FREEZE_ENCODER:
    encoder_lr = config.LSTM_LR * config.ENCODER_LR_RATIO
    print(f"🔓 Fine-tuning encoder (LR={encoder_lr:.6f}) + LSTM + classifier (LR={config.LSTM_LR:.6f})")
    param_groups.append({'params': model.encoder.parameters(), 'lr': encoder_lr, 'name': 'Encoder'})

elif config.USE_AUTOENCODER and config.FREEZE_ENCODER:
    print("🔒 Encoder frozen — training only LSTM + classifier")
else:
    print("ℹ️ No encoder — training LSTM + classifier only")

# Build optimizer
optimizer = optim.Adam(param_groups, weight_decay=1e-4)


# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
early_stopping = EarlyStopping(patience=config.LSTM_PATIENCE, mode='max')


# --- Training Loop ---
print(f"\n🚀 Training for up to {config.LSTM_EPOCHS} epochs with early stopping (patience={config.LSTM_PATIENCE})...")
best_metrics = {}

for epoch in range(config.LSTM_EPOCHS):
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

 
    
    print("-" * 100)
    print(f"Epoch {epoch+1:3d}/{config.LSTM_EPOCHS} | Loss: {avg_loss:.4f} | "
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

    if f1 == 1.0:
        break  # Perfect score achieved

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
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder_state_dict': model.encoder.state_dict() if config.USE_AUTOENCODER else None,
    'num_classes': NUM_CLASSES,
    'label_map': label_map,
    'accuracy': best_metrics["accuracy"],
    'precision': best_metrics["precision"],
    'recall': best_metrics["recall"],
    'f1': best_metrics["f1"],
    'use_pretrained_autoencoder': config.USE_AUTOENCODER,
    'freeze_encoder': config.FREEZE_ENCODER,
    'config': {
        'input_dim': config.INPUT_DIM,
        'latent_dim': config.AE_LATENT_DIM,
        'hidden_dim': config.LSTM_HIDDEN_DIM,
        'lstm_num_layers': config.LSTM_NUM_LAYERS,
        'use_bilstm': config.LSTM_BIDIRECTIONAL,
        'dropout': config.LSTM_DROPOUT,
        'batch_size': config.LSTM_BATCH_SIZE,
        'lr': config.LSTM_LR,
        'encoder_lr_ratio': config.ENCODER_LR_RATIO
    }
}, config.LSTM_BEST)

print(f"\n✅ Best classifier + metrics saved to {config.LSTM_BEST}")
print(f"📈 Confusion matrix saved to confusion_matrix.png")

torch.cuda.empty_cache()