import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import random

# --- Configuration ---
RAW_LANDMARKS_DIR = Path("data/landmarks")
PROCESSED_DIR = Path("data/processed")
SPLIT_RATIOS = {"train": 0.7, "eval": 0.15, "test": 0.15}

EXPECTED_SEQUENCE_LENGTH = 90
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3
EXPECTED_LANDMARKS = NUM_LANDMARKS * LANDMARK_DIMS

# --- ANGLES ---
ANGLES_DIM = 14

JOINT_CHAINS = [
    # --- ARMS (6 angles) ---
    [14, 12, 16],  # Right elbow
    [12, 11, 14],  # Right shoulder
    [13, 11, 15],  # Left elbow
    [11, 12, 13],  # Left shoulder
    [16, 14, 22],  # Right wrist
    [15, 13, 21],  # Left wrist

    # --- LEGS (6 angles) ---
    [26, 24, 28],  # Right knee
    [24, 23, 26],  # Right hip
    [25, 23, 27],  # Left knee
    [23, 24, 25],  # Left hip
    [28, 26, 32],  # Right ankle
    [27, 25, 31],  # Left ankle

    # --- TORSO (2 angles) ---
    [12, 24, 11],  # Torso bend right/left
    [11, 23, 12],  # Torso rotation
]

np.random.seed(1)
random.seed(1)

# -----------------------------
#   FAST ANGLE CALCULATION
# -----------------------------

def vector_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint b with vectors ba and bc."""
    ba = a - b
    bc = c - b

    # Normalize
    ba_n = ba / (np.linalg.norm(ba) + 1e-6)
    bc_n = bc / (np.linalg.norm(bc) + 1e-6)

    cosine = np.dot(ba_n, bc_n)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def landmarks_to_angles(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """Convert landmarks -> 14 angles per frame, vectorized."""
    angle_sequences = []

    for seq in sequences:
        # (N, T, 33, 3)
        seq_reshaped = seq.reshape(seq.shape[0], seq.shape[1], NUM_LANDMARKS, 3)

        # Pre-create empty angle array
        angles = np.zeros((seq.shape[0], seq.shape[1], ANGLES_DIM), dtype=np.float32)

        for sample_idx in range(seq.shape[0]):
            sample = seq_reshaped[sample_idx]

            # Detect padding frames
            zero_mask = np.all(sample == 0, axis=(1, 2))

            for chain_idx, (mid, a, c) in enumerate(JOINT_CHAINS):
                A = sample[:, a]
                B = sample[:, mid]
                C = sample[:, c]

                # Compute angle per frame
                vecBA = A - B
                vecBC = C - B

                BA_norm = vecBA / (np.linalg.norm(vecBA, axis=1, keepdims=True) + 1e-6)
                BC_norm = vecBC / (np.linalg.norm(vecBC, axis=1, keepdims=True) + 1e-6)

                dot = np.clip(np.sum(BA_norm * BC_norm, axis=1), -1, 1)
                angle = np.degrees(np.arccos(dot)) / 180.0  # normalize 0–1

                angles[sample_idx, :, chain_idx] = angle

            # Restore padding frames to zeros
            angles[sample_idx][zero_mask] = 0

        angle_sequences.append(angles)

    return angle_sequences

# -----------------------------
#       DATA AUGMENTATION
# -----------------------------

def add_gaussian_noise(sequence: np.ndarray, std_dev: float = 0.001) -> np.ndarray:
    return np.clip(sequence + np.random.normal(0, std_dev, sequence.shape), 0, 1)


def time_warp(sequence: np.ndarray) -> np.ndarray:
    T = sequence.shape[0]

    speed = random.choice([1.0, random.uniform(0.8, 0.9), random.uniform(1.1, 1.2)])
    if speed == 1.0:
        return sequence

    target_len = int(T / speed)
    idx_old = np.linspace(0, T - 1, T)
    idx_new = np.linspace(0, T - 1, target_len)

    warped = np.vstack([
        np.interp(idx_new, idx_old, sequence[:, i])
        for i in range(sequence.shape[1])
    ]).T

    # Normalize length
    if warped.shape[0] > EXPECTED_SEQUENCE_LENGTH:
        warped = warped[:EXPECTED_SEQUENCE_LENGTH]
    elif warped.shape[0] < EXPECTED_SEQUENCE_LENGTH:
        pad = np.tile(warped[-1], (EXPECTED_SEQUENCE_LENGTH - warped.shape[0], 1))
        warped = np.vstack([warped, pad])

    return warped


def random_frame_dropout(sequence: np.ndarray, max_frames: int = 3) -> np.ndarray:
    valid = np.where(np.any(sequence != 0, axis=1))[0]
    if len(valid) == 0:
        return sequence

    drop_n = min(len(valid), random.randint(1, max_frames))
    drop_idx = np.random.choice(valid, drop_n, replace=False)
    sequence[drop_idx] = 0
    return sequence


def augment_sequences(angle_sequences: List[np.ndarray], volume_multiplier: int = 2) -> List[np.ndarray]:
    augmented = []

    for seq in angle_sequences:
        n_samples = seq.shape[0]

        for i in range(n_samples):
            base = seq[i]

            for _ in range(volume_multiplier):
                s = base.copy()
                s = add_gaussian_noise(s)
                s = time_warp(s)
                s = random_frame_dropout(s)

                augmented.append(s.reshape(1, EXPECTED_SEQUENCE_LENGTH, ANGLES_DIM))

    return [np.concatenate(augmented, axis=0)] if augmented else []


# -----------------------------
#           LOADING
# -----------------------------

def stratified_video_split(video_paths: List[Path]):
    n = len(video_paths)
    if n == 0:
        return [], [], []

    if n < 3:
        warnings.warn(f"Only {n} videos found. Using first for all splits.")
        return [video_paths[0]], [], []

    n_train = max(1, int(n * SPLIT_RATIOS["train"]))
    n_eval = max(1, int(n * SPLIT_RATIOS["eval"]))
    n_test = max(1, n - n_train - n_eval)

    shuffled = np.random.permutation(video_paths).tolist()
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_eval],
        shuffled[n_train + n_eval:n_train + n_eval + n_test]
    )


def validate_sequence_shape(data: np.ndarray, path: Path):
    if data.ndim != 3:
        warnings.warn(f"Bad ndim in {path.name}")
        return False
    if data.shape[1] != EXPECTED_SEQUENCE_LENGTH:
        warnings.warn(f"Bad seq length in {path.name}")
        return False
    if data.shape[2] != EXPECTED_LANDMARKS:
        warnings.warn(f"Bad feature size in {path.name}")
        return False
    return True


def load_and_validate_sequences(paths: List[Path]):
    out = []
    for p in paths:
        try:
            arr = np.load(p)
            if arr.size == 0:
                continue
            if validate_sequence_shape(arr, p):
                out.append(arr)
        except Exception as e:
            warnings.warn(f"Error loading {p.name}: {e}")
    return out


# -----------------------------
#          PROCESS SPLIT
# -----------------------------

def process_split(video_paths, name, split, label, outdir):
    if not video_paths:
        return {"sequences": 0, "samples": 0}

    sequences = load_and_validate_sequences(video_paths)
    if not sequences:
        return {"sequences": 0, "samples": 0}

    angle_sequences = landmarks_to_angles(sequences)

    if split == "train":
        before = sum(s.shape[0] for s in angle_sequences)
        angle_sequences.extend(augment_sequences(angle_sequences, volume_multiplier=2))
        after = sum(s.shape[0] for s in angle_sequences)
        print(f"✨ Augmentation {before} → {after}")

    X = np.concatenate(angle_sequences, axis=0)
    y = np.full(X.shape[0], label, dtype=np.int64)

    save_dir = outdir / name / split
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / f"{name}_{split}.npz", features=X, labels=y)

    print(f"  {split}: {X.shape[0]} samples ({ANGLES_DIM} features)")
    return {"sequences": len(video_paths), "samples": len(X)}


# -----------------------------
#               MAIN
# -----------------------------

def main():
    print("=" * 70)
    print("Preprocessing Pipeline (14 Angles + Augmentation)")
    print("=" * 70)

    if not RAW_LANDMARKS_DIR.exists():
        raise FileNotFoundError(RAW_LANDMARKS_DIR)

    exercise_dirs = sorted([d for d in RAW_LANDMARKS_DIR.iterdir() if d.is_dir()])
    if not exercise_dirs:
        raise ValueError("No exercise folders found")

    label_map = {ex.name: i for i, ex in enumerate(exercise_dirs)}
    label_names = list(label_map.keys())

    print("\nClasses:")
    for i, n in enumerate(label_names):
        print(f"  {i}: {n}")

    # Splits
    splits = {}
    print("\nStep 1: Splitting")
    for ex in exercise_dirs:
        files = sorted(ex.glob("*.npy"))
        train, ev, test = stratified_video_split(files)
        splits[ex.name] = {"train": train, "eval": ev, "test": test}

    PROCESSED_DIR.mkdir(exist_ok=True)
    np.save(PROCESSED_DIR / "label_map.npy", np.array(label_names, dtype=object))

    print("\nStep 2: Processing")
    totals = {s: {"sequences": 0, "samples": 0} for s in ["train", "eval", "test"]}

    for name, sp in splits.items():
        print(f"\n{name}:")
        label = label_map[name]
        for split in ["train", "eval", "test"]:
            stats = process_split(sp[split], name, split, label, PROCESSED_DIR)
            totals[split]["sequences"] += stats["sequences"]
            totals[split]["samples"] += stats["samples"]

    print("\nSummary:")
    for s in ["train", "eval", "test"]:
        print(f"  {s}: {totals[s]['samples']} samples")

    print("\nSaved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
