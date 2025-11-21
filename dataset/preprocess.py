import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import random

# --- Configuration ---
RAW_LANDMARKS_DIR = Path("data/landmarks")
PROCESSED_DIR = Path("data/processed")
SPLIT_RATIOS = {"train": 0.7, "eval": 0.15, "test": 0.15}
EXPECTED_SEQUENCE_LENGTH = 90  # Should match extraction script
NUM_LANDMARKS = 33     # New constant for clarity
LANDMARK_DIMS = 3      # New constant for clarity
EXPECTED_LANDMARKS = NUM_LANDMARKS * LANDMARK_DIMS  # 33 landmarks × 3 coordinates (x,y,z)

# --- NEW ANGLE CONSTANTS ---
ANGLES_DIM = 14  # The actual number of calculated joint angles

# Define key joints for angle calculation (MediaPipe Indices)
# Format: [Anchor/Middle Joint, Start Joint, End Joint]
JOINT_CHAINS = [
    # Right Arm
    [14, 12, 16],  # Elbow Angle (14:Elbow, 12:Shoulder, 16:Wrist)
    [12, 11, 14],  # Shoulder Angle (R) (12:Shoulder, 11:L-Shoulder, 14:Elbow)
    # Left Arm
    [13, 11, 15],  # Elbow Angle (13:Elbow, 11:Shoulder, 15:Wrist)
    [11, 12, 13],  # Shoulder Angle (L) (11:Shoulder, 12:R-Shoulder, 13:Elbow)
    # Right Leg
    [26, 24, 28],  # Knee Angle (26:Knee, 24:Hip, 28:Ankle)
    [24, 23, 26],  # Hip Angle (R) (24:Hip, 23:L-Hip, 26:Knee)
    # Left Leg
    [25, 23, 27],  # Knee Angle (25:Knee, 23:Hip, 27:Ankle)
    [23, 24, 25],  # Hip Angle (L) (23:Hip, 24:R-Hip, 25:Knee)
    # Torso/Back (using midpoint for better twist)
    [12, 24, 11],  # Torso Tilt/Lean (Shoulder (R) -> Hip (R) -> Shoulder (L))
    [11, 23, 12],  # Torso Tilt/Lean (Shoulder (L) -> Hip (L) -> Shoulder (R))
    # Ankle (Simpler 3-point chains for the feet)
    [28, 26, 30],  # Ankle Angle (R)
    [27, 25, 29],  # Ankle Angle (L)
    # Wrist (Simple 3-point chains for the hands)
    [16, 14, 20],  # Wrist Angle (R)
    [15, 13, 19],  # Wrist Angle (L)
]

np.random.seed(1)
random.seed(1) # For data augmentation


# --- ANGLE CALCULATION HELPERS ---

def vector_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates the angle (in degrees) at joint B formed by vectors BA and BC."""
    
    # Vector BA
    ba = a - b
    # Vector BC
    bc = c - b
    
    # Normalize vectors to avoid scaling issues
    # Added 1e-6 to prevent division by zero for zero-length vectors
    ba_norm = ba / (np.linalg.norm(ba) + 1e-6)
    bc_norm = bc / (np.linalg.norm(bc) + 1e-6)

    # Use dot product definition of angle
    cosine_angle = np.dot(ba_norm, bc_norm)
    
    # Clamp to [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Return angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def landmarks_to_angles(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Converts raw landmark sequences into 14 joint angle sequences (no padding).
    
    Args:
        sequences: List of video sequences, each shape (N_samples, Seq_len, 99).
        
    Returns:
        List of angle sequences, each shape (N_samples, Seq_len, 14).
    """
    angle_sequences = []

    for seq in sequences:
        # Reshape to (N_samples, Seq_len, 33, 3)
        reshaped_data = seq.reshape(seq.shape[0], seq.shape[1], NUM_LANDMARKS, LANDMARK_DIMS)
        
        all_samples_angles = []

        for sample_seq in reshaped_data: # sample_seq shape (Seq_len, 33, 3)
            sample_angles = []
            
            for frame in sample_seq: # frame shape (33, 3)
                # Check for zero frame (padding)
                if np.all(frame == 0):
                    # Maintain the padding shape (14 features)
                    sample_angles.append(np.zeros(ANGLES_DIM, dtype=np.float32))
                    continue

                frame_angles = []
                
                for anchor_idx, start_idx, end_idx in JOINT_CHAINS:
                    # Get the landmark coordinates
                    a, b, c = frame[start_idx], frame[anchor_idx], frame[end_idx]
                    
                    # Calculate angle
                    angle = vector_angle(a, b, c)
                    
                    # Store angle normalized by the max possible angle (180 degrees)
                    frame_angles.append(angle / 180.0) 
                    
                # NO PADDING! We use the 14 calculated angles directly.
                sample_angles.append(np.array(frame_angles, dtype=np.float32))
            
            # Stack all frames for this sample
            all_samples_angles.append(np.stack(sample_angles, axis=0))

        # Concatenate and reshape back to (N_samples, Seq_len, 14)
        angle_video_data = np.concatenate(all_samples_angles, axis=0)
        angle_sequences.append(angle_video_data.reshape(seq.shape[0], seq.shape[1], ANGLES_DIM))
            
    return angle_sequences
# --------------------------------------------------


# --- DATA AUGMENTATION FUNCTIONS (UPDATED FOR ANGLES) ---

def add_gaussian_noise(sequence: np.ndarray, std_dev: float = 0.001) -> np.ndarray:
    """
    Adds small random Gaussian noise to the normalized angle features.
    
    std_dev is small (0.001) because the features are scaled between 0.0 and 1.0.
    """
    noise = np.random.normal(0, std_dev, sequence.shape).astype(np.float32)
    
    noisy_sequence = sequence + noise
    
    # Clip to ensure augmented angles stay within the normalized [0.0, 1.0] range.
    return np.clip(noisy_sequence, 0.0, 1.0)

def time_warp(sequence: np.ndarray) -> np.ndarray:
    """Randomly speeds up (dropout) or slows down (interpolation) the sequence."""
    
    sequence_len = sequence.shape[0]
    
    speed_factor = random.choice([
        1.0, 
        random.uniform(0.8, 0.9),  
        random.uniform(1.1, 1.2)
    ])
    
    if speed_factor == 1.0:
        return sequence

    target_len = int(sequence_len / speed_factor)
    
    original_indices = np.linspace(0, sequence_len - 1, sequence_len)
    new_indices = np.linspace(0, sequence_len - 1, target_len)
    
    warped_sequence = np.zeros((target_len, sequence.shape[1]), dtype=np.float32)
    for i in range(sequence.shape[1]):
        warped_sequence[:, i] = np.interp(
            new_indices, original_indices, sequence[:, i]
        )

    # Pad or truncate back to the expected length
    if warped_sequence.shape[0] > EXPECTED_SEQUENCE_LENGTH:
        return warped_sequence[:EXPECTED_SEQUENCE_LENGTH, :]
    elif warped_sequence.shape[0] < EXPECTED_SEQUENCE_LENGTH:
        padding_len = EXPECTED_SEQUENCE_LENGTH - warped_sequence.shape[0]
        padding = np.tile(warped_sequence[-1, :], (padding_len, 1))
        return np.vstack([warped_sequence, padding])
    else:
        return warped_sequence

def random_frame_dropout(sequence: np.ndarray, max_frames: int = 3) -> np.ndarray:
    """Randomly set 1 to max_frames frames to zeros (simulating occlusion)."""
    
    num_drop = random.randint(1, max_frames)
    
    # Only drop from non-zero frames to avoid dropping padding/empty sequences
    valid_indices = np.where(np.any(sequence != 0, axis=1))[0]
    
    if len(valid_indices) < num_drop:
        return sequence
        
    drop_indices = random.sample(valid_indices.tolist(), num_drop)
    
    sequence[drop_indices, :] = 0
    return sequence

def augment_sequences(angle_sequences: List[np.ndarray], volume_multiplier: int = 2) -> List[np.ndarray]:
    """Applies augmentation to angle sequences and increases data volume."""
    newly_augmented_sequences = []
    
    for seq in angle_sequences:
        # seq shape: (n_samples, seq_len, n_features [14])
        n_features = seq.shape[2] 
        
        for sample_idx in range(seq.shape[0]):
            original_sample = seq[sample_idx].copy()
            
            for _ in range(volume_multiplier):
                augmented_sample = original_sample.copy()
                
                # Apply pipeline
                augmented_sample = add_gaussian_noise(augmented_sample)
                augmented_sample = time_warp(augmented_sample)
                augmented_sample = random_frame_dropout(augmented_sample)
                
                # Dynamic reshape 
                newly_augmented_sequences.append(
                    augmented_sample.reshape(1, EXPECTED_SEQUENCE_LENGTH, n_features)
                )
            
    if newly_augmented_sequences:
        return [np.concatenate(newly_augmented_sequences, axis=0)]
        
    return []

# --------------------------------------------------
# (The rest of the utility functions are standard)
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
    """Validate that loaded sequence has expected shape (99 features)."""
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


def process_split(
    video_paths: List[Path],
    exercise_name: str,
    split_name: str,
    label: int,
    output_dir: Path
) -> Dict[str, any]:
    """Process a single data split, applying ANGLE feature extraction."""
    
    if not video_paths:
        return {"sequences": 0, "samples": 0}
    
    # 1. Load data (raw landmarks)
    sequences = load_and_validate_sequences(video_paths)
    
    if not sequences:
        warnings.warn(f"No valid sequences for {exercise_name}/{split_name}")
        return {"sequences": 0, "samples": 0}
    
    # 2. CORE CHANGE: Extract Angles (features are now 14)
    angle_sequences = landmarks_to_angles(sequences)
    
    # 3. AUGMENTATION: Only applied to TRAIN split 
    if split_name == "train":
        initial_sample_count = sum(s.shape[0] for s in angle_sequences)
        # Augmentation creates duplicates and adds them to the list
        angle_sequences.extend(augment_sequences(angle_sequences, volume_multiplier=3))
        print(f"  ✨ Augmentation: {initial_sample_count} -> {sum(s.shape[0] for s in angle_sequences)} samples")
    
    # 4. Concatenate all sequences
    X = np.concatenate(angle_sequences, axis=0)
    y = np.full(len(X), label, dtype=np.int64)
    
    # 5. Save
    split_dir = output_dir / exercise_name / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = split_dir / f"{exercise_name}_{split_name}.npz"
    np.savez_compressed(output_path, features=X, labels=y)
    
    print(f"  ✅ {split_name:5s}: {X.shape[0]:4d} samples from {len(video_paths)} videos. (Features: {ANGLES_DIM} Angles)")
    
    return {"sequences": len(video_paths), "samples": len(X)}


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 70)
    print("Exercise Video Preprocessing Pipeline (14 Angle Features + Augmentation)")
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
        print(f"  {label}: {name}")
    
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
        print(f"  Total videos: {len(npy_files)}")
        print(f"  Train: {len(train_vids)}, Eval: {len(eval_vids)}, Test: {len(test_vids)}")
    
    # Save label map
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "label_map.npy", np.array(label_names, dtype=object))
    print(f"✅ Label mapping saved to: {PROCESSED_DIR / 'label_map.npy'}")
    
    # Step 2: Process all splits
    print(f"\n{'-' * 70}")
    print("Step 2: Processing, ANGLE Extraction, and Augmenting (Train Only) all splits")
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
        print(f"  {split_name.capitalize():5s}: {stats['samples']:5d} samples from {stats['sequences']:3d} videos")
    
    print(f"\n📁 Processed data saved to: {PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()