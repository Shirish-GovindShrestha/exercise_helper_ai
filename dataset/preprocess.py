import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

# --- Configuration ---
RAW_LANDMARKS_DIR = Path("data/landmarks")
PROCESSED_DIR = Path("data/processed")
SPLIT_RATIOS = {"train": 0.7, "eval": 0.15, "test": 0.15}
EXPECTED_SEQUENCE_LENGTH = 60  # Should match extraction script
EXPECTED_LANDMARKS = 33 * 3  # 33 landmarks × 3 coordinates (x,y,z)
np.random.seed(42)


def stratified_video_split(
    video_paths: List[Path]
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split video paths into train/eval/test sets.
    
    Ensures no data leakage by keeping each video in only one split.
    Requires minimum 3 videos for proper splitting.
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


def compute_normalization_stats(train_sequences: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from training sequences.
    
    Args:
        train_sequences: List of arrays with shape (n_sequences, seq_length, n_features)
    
    Returns:
        mean: Array of shape (n_features,)
        std: Array of shape (n_features,)
    """
    if not train_sequences:
        raise ValueError("No training sequences provided for normalization")
    
    # Concatenate all sequences: (total_frames, n_features)
    all_frames = np.concatenate(
        [seq.reshape(-1, seq.shape[-1]) for seq in train_sequences],
        axis=0
    )
    
    print(f"Computing normalization from {all_frames.shape[0]:,} training frames...")
    
    # Compute statistics
    mean = np.mean(all_frames, axis=0, dtype=np.float32)
    std = np.std(all_frames, axis=0, dtype=np.float32)
    
    # Prevent division by zero
    std = np.where(std < 1e-7, 1.0, std)
    
    return mean, std


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


def normalize_sequences(
    sequences: List[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray
) -> List[np.ndarray]:
    """Apply z-score normalization to sequences."""
    normalized = []
    
    for seq in sequences:
        # Broadcasting: (n_seq, seq_len, features) - (features,) / (features,)
        norm_seq = (seq - mean) / std
        normalized.append(norm_seq.astype(np.float32))
    
    return normalized


def process_split(
    video_paths: List[Path],
    exercise_name: str,
    split_name: str,
    label: int,
    mean: np.ndarray,
    std: np.ndarray,
    output_dir: Path
) -> Dict[str, any]:
    """Process a single data split."""
    
    if not video_paths:
        return {"sequences": 0, "samples": 0}
    
    # Load and validate
    sequences = load_and_validate_sequences(video_paths)
    
    if not sequences:
        warnings.warn(f"No valid sequences for {exercise_name}/{split_name}")
        return {"sequences": 0, "samples": 0}
    
    # Normalize
    normalized = normalize_sequences(sequences, mean, std)
    
    # Concatenate all sequences
    X = np.concatenate(normalized, axis=0)
    y = np.full(len(X), label, dtype=np.int64)
    
    # Save
    split_dir = output_dir / exercise_name / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = split_dir / f"{exercise_name}_{split_name}.npz"
    np.savez_compressed(output_path, features=X, labels=y)
    
    print(f"  ✅ {split_name:5s}: {X.shape[0]:4d} samples from {len(sequences)} videos")
    
    return {"sequences": len(sequences), "samples": len(X)}


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 70)
    print("Exercise Video Preprocessing Pipeline")
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
    
    # Step 1: Split videos and collect training data
    print(f"\n{'-' * 70}")
    print("Step 1: Splitting videos into train/eval/test sets")
    print("-" * 70)
    
    per_exercise_splits = {}
    all_train_sequences = []
    
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
        
        # Load training sequences for normalization
        train_seqs = load_and_validate_sequences(train_vids)
        all_train_sequences.extend(train_seqs)
    
    if not all_train_sequences:
        raise RuntimeError("No valid training sequences found!")
    
    # Step 2: Compute normalization statistics
    print(f"\n{'-' * 70}")
    print("Step 2: Computing normalization statistics from training data")
    print("-" * 70)
    
    mean, std = compute_normalization_stats(all_train_sequences)
    
    # Save metadata
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED_DIR / "normalization_stats.npz",
        mean=mean,
        std=std
    )
    np.save(PROCESSED_DIR / "label_map.npy", np.array(label_names, dtype=object))
    
    print(f"✅ Normalization stats: mean={mean.mean():.4f}, std={std.mean():.4f}")
    
    # Step 3: Process all splits
    print(f"\n{'-' * 70}")
    print("Step 3: Processing and normalizing all data splits")
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
                mean,
                std,
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
    print(f"📁 Normalization stats: {PROCESSED_DIR / 'normalization_stats.npz'}")
    print(f"📁 Label mapping: {PROCESSED_DIR / 'label_map.npy'}")
    print("=" * 70)


if __name__ == "__main__":
    main()