import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import random

# --- Configuration ---
RAW_LANDMARKS_DIR = Path("data/landmarks")
PROCESSED_DIR = Path("data/processed")
SPLIT_RATIOS = {"train": 0.7, "eval": 0.15, "test": 0.15}
EXPECTED_SEQUENCE_LENGTH = 60  # Should match extraction script
NUM_LANDMARKS = 33   # New constant for clarity
LANDMARK_DIMS = 3    # New constant for clarity
EXPECTED_LANDMARKS = NUM_LANDMARKS * LANDMARK_DIMS  # 33 landmarks × 3 coordinates (x,y,z)
np.random.seed(1)
random.seed(1) # For data augmentation


# --- BODY-CENTRIC NORMALIZATION FUNCTION (UPDATED) ---
def normalize_body_centric(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Removes global position by translating coordinates to a central anchor point (Mid-Hip or Mid-Shoulder fallback).
    
    Args:
        sequences: List of arrays, each with shape (n_seqs, seq_len, n_features).
        
    Returns:
        List of arrays with body-centric coordinates.
    """
    normalized_sequences = []

    for seq in sequences:
        # 1. Reshape to (N_Sequences, Seq_len, 33, 3)
        reshaped_data = seq.reshape(seq.shape[0], seq.shape[1], NUM_LANDMARKS, LANDMARK_DIMS).copy()
        translated_sequences = []

        # Iterate over each sample sequence (in case video file had multiple sequences)
        for sample_seq in reshaped_data:
            translated_frames = []
            
            # Iterate over each frame in the sequence
            for frame_idx in range(sample_seq.shape[0]):
                frame = sample_seq[frame_idx] # shape (33, 3)
                
                # Check for zero frame (entirely missed)
                if np.all(frame == 0):
                    translated_frames.append(frame) # Keep as (0,0,0) centered
                    continue

                # Define coordinate slices for checking/calculation
                # Left Hip (23), Right Hip (24)
                lh_coords = frame[23, :] 
                rh_coords = frame[24, :] 
                # Left Shoulder (11), Right Shoulder (12)
                ls_coords = frame[11, :] 
                rs_coords = frame[12, :] 
                
                # --- Determine Anchor Point ---
                
                # Primary: Mid-Hip (Check if hips are non-zero)
                is_hip_visible = (not np.all(lh_coords == 0)) and (not np.all(rh_coords == 0))
                is_shoulder_visible = (not np.all(ls_coords == 0)) and (not np.all(rs_coords == 0))

                if is_hip_visible:
                    center_point = (lh_coords + rh_coords) / 2
                
                # Fallback: Mid-Shoulder
                elif is_shoulder_visible:
                    center_point = (ls_coords + rs_coords) / 2
                
                # Last Resort: (0, 0, 0) - If body is not tracked reliably (e.g. only a few non-hip/shoulder points)
                else:
                    center_point = np.zeros(LANDMARK_DIMS, dtype=np.float32)

                # 2. Translate: Subtract the chosen center point from all landmarks
                translated_frame = frame - center_point
                translated_frames.append(translated_frame)
            
            # Re-stack frames for the sample sequence
            translated_sequences.append(np.stack(translated_frames, axis=0))
        
        # 3. Flatten back to (N_Sequences, Seq_len, 99)
        # Handle cases where multiple sequences might result from one video file
        centered_video_data = np.concatenate(translated_sequences, axis=0)
        
        normalized_sequences.append(centered_video_data.reshape(seq.shape))
        
    return normalized_sequences
# --------------------------------------------------


# --- DATA AUGMENTATION FUNCTIONS (NEW) ---
def add_gaussian_noise(sequence: np.ndarray, std_dev: float = 0.01) -> np.ndarray:
    """Add small random Gaussian noise to all landmarks."""
    noise = np.random.normal(0, std_dev, sequence.shape).astype(np.float32)
    return sequence + noise

def time_warp(sequence: np.ndarray) -> np.ndarray:
    """Randomly speeds up (dropout) or slows down (interpolation) the sequence."""
    
    # Sequence shape: (Seq_len, N_Features)
    sequence_len = sequence.shape[0]
    
    # Decide on speed change (1: no change, <1: speed up, >1: slow down)
    speed_factor = random.choice([
        1.0, # No change
        random.uniform(0.8, 0.9),  # Speed up (0.8 to 0.9 of original speed)
        random.uniform(1.1, 1.2)   # Slow down (1.1 to 1.2 of original speed)
    ])
    
    if speed_factor == 1.0:
        return sequence

    target_len = int(sequence_len / speed_factor)
    
    # Original time indices
    original_indices = np.linspace(0, sequence_len - 1, sequence_len)
    # New time indices to sample from the original sequence
    new_indices = np.linspace(0, sequence_len - 1, target_len)
    
    # Interpolate for each feature
    warped_sequence = np.zeros((target_len, sequence.shape[1]), dtype=np.float32)
    for i in range(sequence.shape[1]):
        warped_sequence[:, i] = np.interp(
            new_indices, original_indices, sequence[:, i]
        )

    # Pad or truncate back to the expected length
    if warped_sequence.shape[0] > EXPECTED_SEQUENCE_LENGTH:
        # Truncate
        return warped_sequence[:EXPECTED_SEQUENCE_LENGTH, :]
    elif warped_sequence.shape[0] < EXPECTED_SEQUENCE_LENGTH:
        # Pad with the last frame
        padding_len = EXPECTED_SEQUENCE_LENGTH - warped_sequence.shape[0]
        padding = np.tile(warped_sequence[-1, :], (padding_len, 1))
        return np.vstack([warped_sequence, padding])
    else:
        return warped_sequence

def random_frame_dropout(sequence: np.ndarray, max_frames: int = 3) -> np.ndarray:
    """Randomly set 1 to max_frames frames to zeros."""
    
    num_drop = random.randint(1, max_frames)
    
    # Only drop from non-zero frames to avoid dropping padding/empty sequences
    valid_indices = np.where(np.any(sequence != 0, axis=1))[0]
    
    if len(valid_indices) < num_drop:
        # If sequence is too short or mostly zeros, don't drop frames
        return sequence
        
    drop_indices = random.sample(valid_indices.tolist(), num_drop)
    
    sequence[drop_indices, :] = 0
    return sequence

def augment_sequences(centered_sequences: List[np.ndarray]) -> List[np.ndarray]:
    """Applies a suite of augmentations to training sequences."""
    augmented_sequences = []
    
    for seq in centered_sequences:
        # Seq is (n_sequences, seq_len, n_features)
        
        for sample_idx in range(seq.shape[0]):
            sample = seq[sample_idx] # (seq_len, n_features)
            
            # 1. Gaussian Noise
            augmented_sample = add_gaussian_noise(sample)
            
            # 2. Time Warping
            augmented_sample = time_warp(augmented_sample)
            
            # 3. Random Frame Dropout
            augmented_sample = random_frame_dropout(augmented_sample)
            
            augmented_sequences.append(augmented_sample.reshape(1, EXPECTED_SEQUENCE_LENGTH, EXPECTED_LANDMARKS))
            
    # Concatenate all augmented samples into a single list element
    if augmented_sequences:
        # Combine into one big array for the train set
        augmented_train_data = np.concatenate(augmented_sequences, axis=0) 
        # Return as a list containing the single augmented train array
        return [augmented_train_data]
        
    return []

# --------------------------------------------------
# (The rest of the utility functions remain the same)
# --------------------------------------------------

def stratified_video_split(
    video_paths: List[Path]
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split video paths into train/eval/test sets.
    """
    n = len(video_paths)
    
    if n == 0:
        return [], [], []
    
    if n < 3:
        warnings.warn(
            f"Only {n} video(s) found. Need at least 3 for proper train/eval/test split. "
            f"Using single video for all splits (DATA LEAKAGE WARNING!)"
        )
        return [video_paths[0]], [], []
    
    # Calculate split sizes
    n_train = max(1, int(n * SPLIT_RATIOS["train"]))
    n_eval = max(1, int(n * SPLIT_RATIOS["eval"]))
    n_test = max(1, n - n_train - n_eval)
    
    # Adjust if test would be 0
    if n_test < 1:
        n_test = 1
        n_eval = max(1, n - n_train - n_test)
        n_train = n - n_eval - n_test
    
    # Ensure we don't exceed total videos
    total = n_train + n_eval + n_test
    if total > n:
        n_train = max(1, n_train - (total - n))
    
    # Shuffle and split
    shuffled = np.random.permutation(video_paths).tolist()
    
    train = shuffled[:n_train]
    eval_ = shuffled[n_train:n_train + n_eval]
    test = shuffled[n_train + n_eval:n_train + n_eval + n_test]
    
    return train, eval_, test


def validate_sequence_shape(data: np.ndarray, video_path: Path) -> bool:
    """Validate that loaded sequence has expected shape."""
    if data.ndim != 3:
        warnings.warn(f"Unexpected ndim {data.ndim} in {video_path.name}, expected 3D")
        return False
    
    if data.shape[1] != EXPECTED_SEQUENCE_LENGTH:
        warnings.warn(
            f"Unexpected sequence length {data.shape[1]} in {video_path.name}, "
            f"expected {EXPECTED_SEQUENCE_LENGTH}"
        )
        return False
    
    if data.shape[2] != EXPECTED_LANDMARKS:
        warnings.warn(
            f"Unexpected features {data.shape[2]} in {video_path.name}, "
            f"expected {EXPECTED_LANDMARKS}"
        )
        return False
    
    return True


# NOTE: Removed compute_normalization_stats

def load_and_validate_sequences(
    video_paths: List[Path]
) -> List[np.ndarray]:
    """Load sequences from video paths with validation."""
    sequences = []
    
    for vid_path in video_paths:
        try:
            data = np.load(vid_path)
            
            if data.size == 0:
                warnings.warn(f"Empty array in {vid_path.name}, skipping")
                continue
            
            if not validate_sequence_shape(data, vid_path):
                continue
            
            sequences.append(data)
            
        except Exception as e:
            warnings.warn(f"Error loading {vid_path.name}: {e}")
            continue
    
    return sequences


# NOTE: Removed normalize_sequences


def process_split(
    video_paths: List[Path],
    exercise_name: str,
    split_name: str,
    label: int,
    output_dir: Path
) -> Dict[str, any]:
    """Process a single data split, applying ONLY body-centric normalization."""
    
    if not video_paths:
        return {"sequences": 0, "samples": 0}
    
    # Load and validate
    sequences = load_and_validate_sequences(video_paths)
    
    if not sequences:
        warnings.warn(f"No valid sequences for {exercise_name}/{split_name}")
        return {"sequences": 0, "samples": 0}
    
    # --- Integration: Apply Body-Centric Normalization (Translation) ---
    centered_sequences = normalize_body_centric(sequences)
    
    # --- AUGMENTATION: Only applied to TRAIN split ---
    if split_name == "train":
        initial_sample_count = sum(s.shape[0] for s in centered_sequences)
        centered_sequences.extend(augment_sequences(centered_sequences))
        print(f"  ✨ Augmentation: {initial_sample_count} -> {sum(s.shape[0] for s in centered_sequences)} samples")
    
    # Concatenate all sequences
    X = np.concatenate(centered_sequences, axis=0)
    y = np.full(len(X), label, dtype=np.int64)
    
    # Save
    split_dir = output_dir / exercise_name / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = split_dir / f"{exercise_name}_{split_name}.npz"
    np.savez_compressed(output_path, features=X, labels=y)
    
    print(f"  ✅ {split_name:5s}: {X.shape[0]:4d} samples from {len(video_paths)} videos")
    
    return {"sequences": len(video_paths), "samples": len(X)}


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 70)
    print("Exercise Video Preprocessing Pipeline (Body-Centric Only + Augmentation)")
    print("=" * 70)
    
    # Validate input directory
    if not RAW_LANDMARKS_DIR.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {RAW_LANDMARKS_DIR}")
    
    # Get exercise directories
    exercise_dirs = sorted([d for d in RAW_LANDMARKS_DIR.iterdir() if d.is_dir()])
    
    if not exercise_dirs:
        raise ValueError(f"No exercise folders found in {RAW_LANDMARKS_DIR}")
    
    # Create label mapping
    exercise_to_label = {ex.name: idx for idx, ex in enumerate(exercise_dirs)}
    label_names = [name for name, _ in sorted(exercise_to_label.items(), key=lambda x: x[1])]
    
    print(f"\nFound {len(label_names)} exercise classes:")
    for label, name in enumerate(label_names):
        print(f"  {label}: {name}")
    
    # Step 1: Split videos
    print(f"\n{'-' * 70}")
    print("Step 1: Splitting videos into train/eval/test sets")
    print("-" * 70)
    
    per_exercise_splits = {}
    
    for ex_dir in exercise_dirs:
        npy_files = sorted(list(ex_dir.glob("*.npy")))
        
        if not npy_files:
            warnings.warn(f"No .npy files found in {ex_dir.name}")
            continue
        
        train_vids, eval_vids, test_vids = stratified_video_split(npy_files)
        
        per_exercise_splits[ex_dir.name] = {
            "train": train_vids,
            "eval": eval_vids,
            "test": test_vids
        }
        
        print(f"\n{ex_dir.name}:")
        print(f"  Total videos: {len(npy_files)}")
        print(f"  Train: {len(train_vids)}, Eval: {len(eval_vids)}, Test: {len(test_vids)}")
    
    # Save label map
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "label_map.npy", np.array(label_names, dtype=object))
    print(f"✅ Label mapping saved to: {PROCESSED_DIR / 'label_map.npy'}")

    
    # NOTE: Removed Step 1.5 (centering train data for stats) and Step 2 (compute stats)
    
    # Step 3: Process all splits
    print(f"\n{'-' * 70}")
    print("Step 2: Processing, Body-Centric Normalizing, and Augmenting (Train Only) all splits")
    print("-" * 70)
    
    total_stats = {
        "train": {"sequences": 0, "samples": 0},
        "eval": {"sequences": 0, "samples": 0},
        "test": {"sequences": 0, "samples": 0}
    }
    
    for exercise_name, splits in per_exercise_splits.items():
        label = exercise_to_label[exercise_name]
        print(f"\n{exercise_name} (label={label}):")
        
        for split_name in ["train", "eval", "test"]:
            video_list = splits[split_name]
            
            stats = process_split(
                video_list,
                exercise_name,
                split_name,
                label,
                PROCESSED_DIR
            )
            
            total_stats[split_name]["sequences"] += stats["sequences"]
            total_stats[split_name]["samples"] += stats["samples"]
    
    # Summary
    print(f"\n{'=' * 70}")
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\nDataset Summary:")
    for split_name in ["train", "eval", "test"]:
        stats = total_stats[split_name]
        print(f"  {split_name.capitalize():5s}: {stats['samples']:5d} samples from {stats['sequences']:3d} videos")
    
    print(f"\n📁 Processed data saved to: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()