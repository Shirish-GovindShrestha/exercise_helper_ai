import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from data_processing.dataset_loader import load_split
from models.autoencoder import FrameAutoencoder
from models.early_stopping import EarlyStopping
import config

# --- Load Data ---
print("\n" + "="*70)
print("Loading data...")
print("="*70)

X_train, _ = load_split("train", mode=config.INPUT_FEATURE)
X_eval, _ = load_split("eval", mode=config.INPUT_FEATURE)

print(f"Train samples: {len(X_train):,}")
print(f"Eval samples: {len(X_eval):,}")

X_train_t = torch.from_numpy(X_train).float()
X_eval_t = torch.from_numpy(X_eval).float()

train_loader = DataLoader(TensorDataset(X_train_t), batch_size=config.AE_BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(TensorDataset(X_eval_t), batch_size=config.AE_BATCH_SIZE, shuffle=False)


# --- Initialize Model ---
print("\n" + "="*70)
print("Initializing autoencoder model...")
print("="*70)

model = FrameAutoencoder(
    input_dim=config.INPUT_DIM,
    latent_dim=config.AE_LATENT_DIM,
    dropout=config.AE_DROPOUT
).to(config.DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = optim.Adam(model.parameters(), lr=config.AE_LR)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=config.AE_PATIENCE)


# --- Training ---
history = {"train_loss": [], "val_loss": []}

print("\n" + "="*70)
print(f"🚀 Starting AUTOENCODER training (patience={config.AE_PATIENCE})")
print("="*70)

start_time = time.time()

for epoch in range(config.AE_EPOCHS):
    model.train()
    total_loss = 0

    for (X,) in train_loader:
        X = X.to(config.DEVICE)
        optimizer.zero_grad()

        X_recon, _ = model(X)
        loss = criterion(X_recon, X)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for (X,) in eval_loader:
            X = X.to(config.DEVICE)
            X_recon, _ = model(X)
            total_eval_loss += criterion(X_recon, X).item()

    avg_val_loss = total_eval_loss / len(eval_loader)

    print(f"\nEpoch {epoch+1}/{config.AE_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.6f} | "
          f"Val Loss: {avg_val_loss:.6f}")

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)

    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("🛑 Early stopping triggered.")
        break


# --- Training Complete ---
total_time = time.time() - start_time
metric_name = "loss" if early_stopping.mode == "min" else "score"
print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"⏱️  Total time: {total_time/60:.1f} minutes")
print(f"🏆 Best validation {metric_name}: {early_stopping.best_value:.6f}")


# Load best weights
model.load_state_dict(early_stopping.best_model_state)

# Save autoencoder
torch.save({
    "model_state_dict": model.state_dict(),
    "latent_dim": config.AE_LATENT_DIM,
    "history": history,
}, config.AE_BEST)

print(f"\n✅ Best AUTOENCODER saved to {config.AE_BEST}")
