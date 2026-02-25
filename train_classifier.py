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
from models.lstm import ExerciseClassifier
import config


DEVICE = config.DEVICE


# --- Load Data ---
print("📦 Loading data...")
label_map = get_label_map()
NUM_CLASSES = len(label_map)

X_train, y_train = load_split("train", mode=config.INPUT_FEATURE)
X_eval, y_eval = load_split("eval", mode=config.INPUT_FEATURE)
X_test, y_test = load_split("test", mode=config.INPUT_FEATURE)

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
X_eval_t = torch.from_numpy(X_eval).float()
y_eval_t = torch.from_numpy(y_eval).long()
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).long()



train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=config.GRU_BATCH_SIZE,
    shuffle=True
)
eval_loader = DataLoader(
    TensorDataset(X_eval_t, y_eval_t),
    batch_size=config.GRU_BATCH_SIZE,
    shuffle=False
)

# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

print(f"✅ Loaded {len(X_train)} training samples, {len(X_eval)} eval samples, {len(X_test)} test samples")
print(f"📊 Number of classes: {NUM_CLASSES}")
print(f"📏 Input dimension: {X_train.shape[-1]}")

print(f"\n🏗️ Building GRU classifier...")
model = ExerciseClassifier(
    num_classes=NUM_CLASSES,
    input_dim=X_train.shape[-1],
    hidden_dim=config.GRU_HIDDEN_DIM,
    lstm_num_layers=config.GRU_NUM_LAYERS,
    use_bilstm=config.GRU_BIDIRECTIONAL,
    dropout=config.GRU_DROPOUT
).to(DEVICE)


# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Build optimizer
optimizer = optim.Adam(model.parameters(), lr=config.GRU_LR, weight_decay=config.GRU_WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GRU_SCHEDULER_GAMMA)


# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
early_stopping = EarlyStopping(patience=config.GRU_PATIENCE, mode='max')


# --- Training Loop ---
print(f"\n🚀 Training for up to {config.GRU_EPOCHS} epochs with early stopping (patience={config.GRU_PATIENCE})...")
best_metrics = {}

for epoch in range(config.GRU_EPOCHS):
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

 
    scheduler.step()

    print("-" * 100)
    print(f"Epoch {epoch+1:3d}/{config.GRU_EPOCHS} | Loss: {avg_loss:.4f} | "
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


# --- Final Evaluation on Test Set ---
print("\n" + "="*60)
print("🏁 Final Evaluation on Test Set (Best Model):")
print("="*60)

model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for i in range(0, len(X_test_t), config.GRU_BATCH_SIZE):
        batch_X = X_test_t[i:i+config.GRU_BATCH_SIZE].to(DEVICE)
        batch_y = y_test_t[i:i+config.GRU_BATCH_SIZE]
        
        logits = model(batch_X)
        preds = logits.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(batch_y.numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# Compute test metrics
test_acc = np.mean(test_preds == test_labels)
test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")
print("="*60)

# Get class names
if isinstance(label_map, dict):
    class_names = list(label_map.keys())
else:
    class_names = list(label_map)

# Confusion Matrix (using test data)
cm = confusion_matrix(test_labels, test_preds)
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

# Classification Report (using test data)
print("\n📊 Classification Report (Test Set):")
print(classification_report(
    test_labels,
    test_preds,
    target_names=class_names,
    digits=4
))

# Save model with test metrics
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': NUM_CLASSES,
    'label_map': label_map,
    'val_accuracy': best_metrics["accuracy"],
    'val_precision': best_metrics["precision"],
    'val_recall': best_metrics["recall"],
    'val_f1': best_metrics["f1"],
    'test_accuracy': test_acc,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'config': {
        'input_dim': X_train.shape[-1],
        'hidden_dim': config.GRU_HIDDEN_DIM,
        'lstm_num_layers': config.GRU_NUM_LAYERS,
        'use_bilstm': config.GRU_BIDIRECTIONAL,
        'dropout': config.GRU_DROPOUT,
        'batch_size': config.GRU_BATCH_SIZE,
        'lr': config.GRU_LR
    }
}, config.GRU_BEST)

print(f"\n✅ Best classifier + metrics saved to {config.GRU_BEST}")
print(f"📈 Confusion matrix saved to confusion_matrix.png")

torch.cuda.empty_cache()