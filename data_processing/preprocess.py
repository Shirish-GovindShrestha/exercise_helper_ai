import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import random

# --- Configuration ---
RAW_LANDMARKS_DIR = Path("data/landmarks")
PROCESSED_DIR = Path("data/processed")
SPLIT_RATIOS = {"train": 0.7, "eval": 0.15, "test": 0.15}

EXPECTED_SEQUENCE_LENGTH = 75
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3
EXPECTED_LANDMARKS = NUM_LANDMARKS * LANDMARK_DIMS

# --- ANGLES (12 angles) ---
ANGLES_DIM = 12
TOTAL_FEATURES = ANGLES_DIM + EXPECTED_LANDMARKS  # 12 + 99 = 111

JOINT_CHAINS = [
    # --- ARMS (4 angles) ---
    [12, 14, 16],  # Right shoulder angle
    [11, 13, 15],  # Left shoulder angle
    [14, 12, 16],  # Right elbow angle
    [13, 11, 15],  # Left elbow angle

    # --- LEGS (6 angles) ---
    [24, 23, 26],  # Right hip angle
    [23, 24, 25],  # Left hip angle
    [26, 24, 28],  # Right knee angle
    [25, 23, 27],  # Left knee angle
    [28, 26, 32],  # Right ankle angle
    [27, 25, 31],  # Left ankle angle

    # --- TORSO (2 angles) ---
    [12, 24, 26],  # Torso inclination (right side)
    [11, 23, 12],  # Spine/neck alignment
]

np.random.seed(42)
random.seed(42)

# -----------------------------
#   FAST ANGLE CALCULATION
# -----------------------------

def landmarks_to_angles_and_landmarks(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """Convert landmarks -> 12 angles + 99 landmarks per frame."""
    combined_sequences = []

    for seq in sequences:
        # (N, T, 99) -> (N, T, 33, 3)
        seq_reshaped = seq.reshape(seq.shape[0], seq.shape[1], NUM_LANDMARKS, 3)

        # Pre-create arrays
        angles = np.zeros((seq.shape[0], seq.shape[1], ANGLES_DIM), dtype=np.float32)
        landmarks_flat = seq  # Keep original (N, T, 99)

        for sample_idx in range(seq.shape[0]):
            sample = seq_reshaped[sample_idx]

            # Detect padding frames
            zero_mask = np.all(sample == 0, axis=(1, 2))

            for chain_idx, (a, mid, c) in enumerate(JOINT_CHAINS):
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

        # Combine: [angles (12) | landmarks (99)] = 111 features
        combined = np.concatenate([angles, landmarks_flat], axis=2)
        combined_sequences.append(combined)

    return combined_sequences

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


def augment_sequences(combined_sequences: List[np.ndarray], volume_multiplier: int = 2) -> List[np.ndarray]:
    augmented = []

    for seq in combined_sequences:
        n_samples = seq.shape[0]

        for i in range(n_samples):
            base = seq[i]

            for _ in range(volume_multiplier):
                s = base.copy()
                s = add_gaussian_noise(s)
                s = time_warp(s)
                s = random_frame_dropout(s)

                augmented.append(s.reshape(1, EXPECTED_SEQUENCE_LENGTH, TOTAL_FEATURES))

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

    combined_sequences = landmarks_to_angles_and_landmarks(sequences)

    if split == "train":
        before = sum(s.shape[0] for s in combined_sequences)
        combined_sequences.extend(augment_sequences(combined_sequences, volume_multiplier=2))
        after = sum(s.shape[0] for s in combined_sequences)
        print(f"✨ Augmentation {before} → {after}")

    X = np.concatenate(combined_sequences, axis=0)
    y = np.full(X.shape[0], label, dtype=np.int64)

    save_dir = outdir / name / split
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / f"{name}_{split}.npz", features=X, labels=y)

    print(f"  {split}: {X.shape[0]} samples ({TOTAL_FEATURES} features: 12 angles + 99 landmarks)")
    return {"sequences": len(video_paths), "samples": len(X)}


# -----------------------------
#               MAIN
# -----------------------------

def main():
    print("=" * 70)
    print("Preprocessing Pipeline (12 Angles + 99 Landmarks = 111 Features)")
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

    print("\nFeature breakdown:")
    print(f"  - Angles: {ANGLES_DIM}")
    print(f"  - Landmarks: {EXPECTED_LANDMARKS}")
    print(f"  - Total: {TOTAL_FEATURES}")
    print("\nSaved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()