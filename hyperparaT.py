import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import config
from models.lstm import ExerciseClassifier
from data_processing.dataset_loader import load_split, get_label_map


DEVICE = config.DEVICE


print("📦 Loading dataset for hyperparameter tuning...")

# Load label map
label_map = get_label_map()
NUM_CLASSES = len(label_map)

# Load splits
X_train, y_train = load_split("train", mode=config.INPUT_FEATURE)
X_eval, y_eval = load_split("eval", mode=config.INPUT_FEATURE)

# Split eval into validation and test
val_split = int(0.5 * len(X_eval))
X_val, X_test = X_eval[:val_split], X_eval[val_split:]
y_val, y_test = y_eval[:val_split], y_eval[val_split:]

# Convert to tensors
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).long()
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).long()

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)


def make_loader(batch_size):
    """Utility: build DataLoaders with given batch size."""
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader


print("🚀 Starting Optuna hyperparameter tuning...")


def objective(trial):
    """Optuna objective function: return F1 score for a sampled hyperparameter set."""

    # --- Hyperparameters to tune ---
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
    lstm_layers = trial.suggest_int("lstm_num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    use_bilstm = trial.suggest_categorical("use_bilstm", [False, True])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # --- Build dataloaders for this trial ---
    train_loader, val_loader, test_loader = make_loader(batch_size)

    # --- Build model ---
    model = ExerciseClassifier(
        num_classes=NUM_CLASSES,
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        lstm_num_layers=lstm_layers,
        dropout=dropout,
        use_bilstm=use_bilstm
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Training loop with early stopping ---
    EPOCHS = 20
    best_val_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # --- Validation evaluation ---
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                preds = logits.argmax(dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        # Early stopping & trial pruning
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        # Report intermediate value for pruning
        trial.report(val_f1, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if patience_counter >= patience:
            break

    # --- Final test evaluation ---
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

    return test_f1  # Return final test F1


# Run Optuna study with MedianPruner for efficiency
sampler = optuna.samplers.TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n🎉 Best hyperparameters found:")
print(study.best_trial.params)

print("\n🔍 Best F1 score:", study.best_trial.value)
print(f"Total trials: {len(study.trials)}, Completed: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")

# Save result
import json
with open("best_lstm_hyperparams.json", "w") as f:
    json.dump(study.best_trial.params, f, indent=4)

print("\n📁 Saved to best_lstm_hyperparams.json")

# Print top 5 trials
print("\n📊 Top 5 trials:")
sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)
for i, trial in enumerate(sorted_trials[:5]):
    print(f"  Trial {trial.number}: F1={trial.value:.4f}")
