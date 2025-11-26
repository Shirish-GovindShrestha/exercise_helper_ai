import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import random
import config

# --- Configuration ---
RAW_LANDMARKS_DIR = Path("data/landmarks")
PROCESSED_DIR = Path("data/processed")
SPLIT_RATIOS = {"train": 0.7, "eval": 0.15, "test": 0.15}

EXPECTED_SEQUENCE_LENGTH = config.EXPECTED_SEQUENCE_LENGTH
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3
EXPECTED_LANDMARKS = NUM_LANDMARKS * LANDMARK_DIMS # 99

# --- ANGLES (12 angles) ---
ANGLES_DIM = 12
TOTAL_FEATURES = ANGLES_DIM + EXPECTED_LANDMARKS  # 12 + 99 = 111

# Landmark indices for key joints (based on standard MediaPipe Pose)
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
TORSO_SCALE_EPSILON = 1e-6 # Epsilon to prevent division by zero

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

np.random.seed(5)
random.seed(5)

# ---------------------------------------------
#   RELATIVE LANDMARK NORMALIZATION 🌟
# ---------------------------------------------

def relative_normalize_landmarks(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Applies centering (to Mid-Hip) and scaling (by Torso Length) to landmark sequences.
    Input: List of arrays (N_samples, T, 99). Output: List of normalized arrays.
    """
    normalized_sequences = []

    for seq_array in sequences:
        N, T, F = seq_array.shape
        seq_reshaped = seq_array.reshape(N, T, NUM_LANDMARKS, 3).copy() # (N, T, 33, 3)
        
        # 1. Calculate Center (Mid-Hip)
        # Mid-Hip is the average of landmarks 23 and 24.
        P_mid_hip = (seq_reshaped[..., LEFT_HIP, :] + seq_reshaped[..., RIGHT_HIP, :]) / 2.0 # (N, T, 3)
        
        # 2. Centering: Subtract Mid-Hip from all 33 landmarks
        # Uses broadcasting: (N, T, 33, 3) - (N, T, 1, 3)
        seq_reshaped = seq_reshaped - np.expand_dims(P_mid_hip, axis=2) 
        
        # 3. Calculate Scaling Factor (Torso Length)
        # Mid-Shoulder is the average of landmarks 11 and 12.
        P_mid_shoulder = (seq_reshaped[..., LEFT_SHOULDER, :] + seq_reshaped[..., RIGHT_SHOULDER, :]) / 2.0
        
        # Torso Length (distance between Mid-Shoulder and Mid-Hip - which is now at origin (0,0,0) after centering)
        # We can just use the norm of P_mid_shoulder
        D_scale = np.linalg.norm(P_mid_shoulder, axis=-1, keepdims=True) # (N, T, 1)
        
        # Replace zero distances (for padded/zero frames) with epsilon
        D_scale = np.where(D_scale < TORSO_SCALE_EPSILON, TORSO_SCALE_EPSILON, D_scale)
        
        # 4. Scaling: Divide all coordinates by Torso Length
        # Uses broadcasting: (N, T, 33, 3) / (N, T, 1, 1)
        # We need to reshape D_scale for correct broadcasting to (N, T, 1, 1)
        D_scale_broadcast = np.expand_dims(D_scale, axis=2)
        
        normalized_array = seq_reshaped / D_scale_broadcast
        
        # Reshape back to (N, T, 99)
        normalized_sequences.append(normalized_array.reshape(N, T, F))
        
    return normalized_sequences


# ---------------------------------------------
#   FAST ANGLE CALCULATION (SINGLE SEQUENCE)
# ---------------------------------------------

def calculate_angles_from_sequence(seq: np.ndarray) -> np.ndarray:
    """Calculates angles for a single sequence/sample: (T, 99) -> (T, 12)."""
    T, F = seq.shape
    
    # (T, 99) -> (T, 33, 3)
    seq_reshaped = seq.reshape(T, NUM_LANDMARKS, 3)

    # Pre-create arrays
    angles = np.zeros((T, ANGLES_DIM), dtype=np.float32)
    
    # Detect padding frames
    zero_mask = np.all(seq_reshaped == 0, axis=(1, 2))

    for chain_idx, (a, mid, c) in enumerate(JOINT_CHAINS):
        A = seq_reshaped[:, a]
        B = seq_reshaped[:, mid]
        C = seq_reshaped[:, c]

        # Compute angle per frame
        vecBA = A - B
        vecBC = C - B

        # Normalize vectors for dot product
        BA_norm = vecBA / (np.linalg.norm(vecBA, axis=1, keepdims=True) + 1e-6)
        BC_norm = vecBC / (np.linalg.norm(vecBC, axis=1, keepdims=True) + 1e-6)

        dot = np.clip(np.sum(BA_norm * BC_norm, axis=1), -1, 1)
        # Normalize angle 0-1
        angle = np.degrees(np.arccos(dot)) / 180.0

        angles[:, chain_idx] = angle

    # Restore padding frames to zeros
    angles[zero_mask] = 0

    return angles

def landmarks_to_angles_and_landmarks(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """Convert landmarks -> 12 angles + 99 landmarks per frame, for a list of arrays."""
    combined_sequences = []

    for seq_array in sequences: # seq_array shape is (N_samples, T, 99)
        combined_samples = []
        for seq in seq_array: # seq shape is (T, 99)
            angles = calculate_angles_from_sequence(seq)
            # Combine: [angles (12) | landmarks (99)] = 111 features
            combined = np.concatenate([angles, seq], axis=1) # (T, 111)
            combined_samples.append(combined)
        
        # Stack the combined samples back into one array for the list
        if combined_samples:
            combined_sequences.append(np.stack(combined_samples, axis=0))

    return combined_sequences

# -----------------------------
#       DATA AUGMENTATION
# -----------------------------

def add_gaussian_noise(sequence: np.ndarray, std_dev: float = 0.001) -> np.ndarray:
    """Adds noise only to non-zero frames of the landmark data."""
    valid_mask = np.any(sequence != 0, axis=1, keepdims=True)
    noise = np.random.normal(0, std_dev, sequence.shape)
    
    sequence = sequence + (noise * valid_mask)
    
    # Note: Clipping is no longer to [0, 1] as relative coordinates can be negative
    # We rely on the network learning the range, but we might want to clip to a reasonable range 
    # like [-2, 2] to prevent outliers if the torso length is tiny due to bad tracking.
    return sequence # We avoid clipping to preserve relative values

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
        padding_value = warped[-1] if np.any(warped[-1] != 0) else np.zeros(warped.shape[1])
        pad = np.tile(padding_value, (EXPECTED_SEQUENCE_LENGTH - warped.shape[0], 1))
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


def augment_landmarks(landmark_sequences: List[np.ndarray], volume_multiplier: int = 2) -> List[np.ndarray]:
    """
    Applies augmentation ONLY to the 99 landmark features (T, 99) 
    and returns a list of individual augmented landmark sequences.
    """
    augmented_landmarks = []

    for seq_array in landmark_sequences: # seq_array shape is (N_samples, T, 99)
        n_samples = seq_array.shape[0]

        for i in range(n_samples):
            base_landmarks = seq_array[i] # shape (T, 99)

            for _ in range(volume_multiplier):
                s = base_landmarks.copy()
                
                # Apply augmentation functions directly to the landmark data
                s = add_gaussian_noise(s)
                s = time_warp(s)
                s = random_frame_dropout(s)

                augmented_landmarks.append(s) # Append the (T, 99) array

    return augmented_landmarks

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

def process_split(video_paths, name, split, label, outdir, global_stats: Dict = None):
    if not video_paths:
        return {"sequences": 0, "samples": 0}

    # Load raw landmark sequences (List[np.ndarray] where each array is (N_samples, T, 99))
    raw_landmark_sequences = load_and_validate_sequences(video_paths)
    if not raw_landmark_sequences:
        return {"sequences": 0, "samples": 0}
    
    # --- Relative Normalization Step ---
    # Apply centering and scaling to all splits
    normalized_landmark_sequences = relative_normalize_landmarks(raw_landmark_sequences)
    
    # --- Augmentation & Feature Conversion Step ---
    
    final_sequences_to_process = normalized_landmark_sequences

    if split == "train":
        # Augment ONLY the normalized landmark data (99 features)
        augmented_landmark_samples = augment_landmarks(normalized_landmark_sequences, volume_multiplier=1)
        
        if augmented_landmark_samples:
            augmented_array = np.stack(augmented_landmark_samples, axis=0) # (N_aug, T, 99)
            final_sequences_to_process.append(augmented_array)
            
        before = sum(s.shape[0] for s in raw_landmark_sequences)
        after = sum(s.shape[0] for s in final_sequences_to_process)
        print(f"✨ Augmentation {before} → {after}")

    # Now, convert ALL landmark data (normalized + augmented) into the 111-feature set
    combined_sequences = landmarks_to_angles_and_landmarks(final_sequences_to_process)

    X = np.concatenate(combined_sequences, axis=0)
    y = np.full(X.shape[0], label, dtype=np.int64)

    save_dir = outdir / name / split
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / f"{name}_{split}.npz", features=X, labels=y)

    print(f"  {split}: {X.shape[0]} samples ({TOTAL_FEATURES} features: 12 angles + 99 relative landmarks)")
    return {"sequences": len(video_paths), "samples": len(X)}


# -----------------------------
#               MAIN
# -----------------------------

def main():
    print("=" * 70)
    print("Preprocessing Pipeline (Relative Normalization + 12 Angles = 111 Features)")
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
    
    # Relative normalization doesn't require global stats
    global_stats = {} 

    for name, sp in splits.items():
        print(f"\n{name}:")
        label = label_map[name]
        for split in ["train", "eval", "test"]:
            stats = process_split(sp[split], name, split, label, PROCESSED_DIR, global_stats)
            totals[split]["sequences"] += stats["sequences"]
            totals[split]["samples"] += stats["samples"]

    print("\nSummary:")
    for s in ["train", "eval", "test"]:
        print(f"  {s}: {totals[s]['samples']} samples")

    print("\nFeature breakdown:")
    print(f"  - Angles: {ANGLES_DIM}")
    print(f"  - Landmarks: {EXPECTED_LANDMARKS} (Relatively Normalized)")
    print(f"  - Total: {TOTAL_FEATURES}")
    print("\nSaved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()