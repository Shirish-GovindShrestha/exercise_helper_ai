#!/usr/bin/env python3
"""
train.py
Unified training script that consumes the **new** dataset loader:
  --stage {ae,clf}     train auto-encoder OR classifier
  --freeze             freeze encoder when training classifier
  --ae_ckpt PATH       encoder checkpoint (required for clf)
  --norm               apply saved normalization (mean/std)
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# custom
from config import (DEVICE, INPUT_DIM, AE_BATCH_SIZE, AE_EPOCHS, AE_LR, AE_PATIENCE,
                    LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_LR, LSTM_PATIENCE,
                    AE_BEST, LSTM_BEST)
from models.autoencoder import AutoEncoder
from models.lstm import LSTMClassifier
from dataset.dataset_loader import (load_split, get_label_map, get_normalization_stats,
                                    clear_cache)   # optional


# ---------- early stopper ----------
class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience, self.mode, self.counter = patience, mode, 0
        self.best = None

    def __call__(self, metric: float) -> bool:
        if self.best is None or \
           (metric > self.best and self.mode == "max") or \
           (metric < self.best and self.mode == "min"):
            self.best, self.counter = metric, 0
            return True
        self.counter += 1
        return self.counter >= self.patience


# ---------- tiny wrapper to apply normalization on-the-fly ----------
class NormDataset(Dataset):
    def __init__(self, split: str, apply_norm: bool = False):
        self.X, self.y = load_split(split, shuffle=(split == "train"))
        self.mean, self.std = (None, None)
        if apply_norm:
            mean, std = get_normalization_stats()
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std  = torch.tensor(std,  dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        return x, y


# ---------- generic loader ----------
def get_loaders(stage: str, apply_norm: bool = False):
    """return train / eval DataLoader + num_classes + label_map"""
    train_ds = NormDataset("train", apply_norm)
    eval_ds  = NormDataset("eval",  apply_norm)
    batch_size = AE_BATCH_SIZE if stage == "ae" else LSTM_BATCH_SIZE
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    ev_loader = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False)
    label_map = get_label_map()
    return tr_loader, ev_loader, len(label_map), label_map


# ---------- stage-1: auto-encoder ----------
def train_ae(norm: bool = False):
    print("\n" + "="*70 + "\n🚀  STAGE  Auto-Encoder\n" + "="*70)
    tr_loader, ev_loader, _, _ = get_loaders("ae", norm)
    model = AutoEncoder(INPUT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = torch.nn.MSELoss()
    stopper = EarlyStopper(AE_PATIENCE, mode="min")

    best_loss = float("inf")
    for epoch in range(1, AE_EPOCHS + 1):
        # ---- train ----
        model.train()
        loss_sum = 0.0
        for x, _ in tr_loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = criterion(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
        tr_loss = loss_sum / len(tr_loader.dataset)

        # ---- eval ----
                # ---- eval ----
        model.eval()
        eval_loss_sum, eval_n = 0.0, 0
        with torch.no_grad():
            for x, _ in ev_loader:
                x = x.to(DEVICE)
                recon, _ = model(x)
                # safe MSE
                loss = torch.nn.functional.mse_loss(recon, x, reduction='sum')
                eval_loss_sum += loss.item()
                eval_n += x.size(0)
        ev_loss = eval_loss_sum / max(eval_n, 1)   # avoid / 0

        print(f"Ep {epoch:02d} | tr-loss {tr_loss:.5f} | val-loss {ev_loss:.5f}")
        if stopper(ev_loss):
            print("🛑  AE early-stop triggered"); break
        if ev_loss < best_loss:
            best_loss = ev_loss
            latent_dim = model.encoder[-2].out_features
            torch.save({"model_state_dict": model.state_dict(),
                        "latent_dim": latent_dim}, AE_BEST)
    print(f"✅  Best AE loss {best_loss:.5f}  →  {AE_BEST}")


# ---------- stage-2: LSTM classifier ----------
def train_clf(ae_ckpt: Path, freeze: bool = False, norm: bool = False):
    print("\n" + "="*70 + f"\n🚀  STAGE  LSTM Classifier  –  freeze={freeze}\n" + "="*70)
    tr_loader, ev_loader, num_classes, label_map = get_loaders("clf", norm)

    # ---- load encoder ----
    ckpt = torch.load(ae_ckpt, map_location=DEVICE)
    encoder = AutoEncoder(INPUT_DIM).to(DEVICE)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = not freeze
    latent_dim = ckpt["latent_dim"]

    # ---- LSTM ----
    model = LSTMClassifier(latent_dim, num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    # class-weighted loss
    weights = torch.tensor(torch.bincount(tr_loader.dataset.y), dtype=torch.float, device=DEVICE)
    weights = (weights / weights.sum()) * num_classes
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    stopper = EarlyStopper(LSTM_PATIENCE, mode="max")

    best_f1 = 0.0
    for epoch in range(1, LSTM_EPOCHS + 1):
        # ---- train ----
        model.train()
        loss_sum, n = 0, 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                z = encoder.encode_only(x)
            logits = model(z)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        tr_loss = loss_sum / n

        # ---- eval ----
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in ev_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                z = encoder.encode_only(x)
                preds.append(model(z).argmax(1).cpu())
                labels.append(y.cpu())
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        acc = (preds == labels).mean()
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        print(f"Ep {epoch:02d} | tr-loss {tr_loss:.4f} | acc {acc:.4f} | f1 {f1:.4f}")
        if stopper(f1):
            print("🛑  CLF early-stop triggered"); break
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state_dict": model.state_dict(),
                        "encoder_state_dict": encoder.state_dict() if not freeze else None,
                        "num_classes": num_classes,
                        "label_map": label_map,
                        "latent_dim": latent_dim,
                        "freeze_encoder": freeze}, LSTM_BEST)
    print(f"✅  Best CLF F1 {best_f1:.4f}  →  {LSTM_BEST}")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train AE or LSTM classifier")
    p.add_argument("--stage", choices=["ae", "clf"], required=True,
                   help="ae: auto-encoder only; clf: classifier only")
    p.add_argument("--freeze", action="store_true",
                   help="freeze encoder while training classifier")
    p.add_argument("--ae_ckpt", type=Path, default=AE_BEST,
                   help="encoder checkpoint (required for clf)")
    p.add_argument("--norm", action="store_true",
                   help="apply saved mean/std normalization")
    return p.parse_args()


# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    if args.stage == "ae":
        train_ae(norm=args.norm)
    else:
        if not args.ae_ckpt.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {args.ae_ckpt}")
        train_clf(args.ae_ckpt, freeze=args.freeze, norm=args.norm)