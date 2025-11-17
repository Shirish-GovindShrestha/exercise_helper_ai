import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings

# Configuration
PROCESSED_DIR = Path("data/processed")

# Global cache for metadata
_label_map_cache = None
_label_to_int_cache = None
_norm_stats_cache = None
_data_cache = {}  # Cache for loaded splits


def get_label_map() -> np.ndarray:
    """
    Load and cache the label mapping array.
    
    Returns:
        Array of exercise names corresponding to label indices
    """
    global _label_map_cache
    
    if _label_map_cache is None:
        label_map_path = PROCESSED_DIR / "label_map.npy"
        
        if not label_map_path.exists():
            raise FileNotFoundError(
                f"Label map not found: {label_map_path}\n"
                "Run preprocessing script first."
            )
        
        _label_map_cache = np.load(label_map_path, allow_pickle=True)
        
    return _label_map_cache


def get_label_to_int() -> Dict[str, int]:
    """
    Get mapping from exercise name to integer label.
    
    Returns:
        Dictionary mapping exercise names to labels
    """
    global _label_to_int_cache
    
    if _label_to_int_cache is None:
        label_map = get_label_map()
        _label_to_int_cache = {name: idx for idx, name in enumerate(label_map)}
    
    return _label_to_int_cache


def get_normalization_stats() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and cache normalization statistics.
    
    Returns:
        Tuple of (mean, std) arrays
    """
    global _norm_stats_cache
    
    if _norm_stats_cache is None:
        # Try new format first
        stats_path = PROCESSED_DIR / "normalization_stats.npz"
        
        if stats_path.exists():
            data = np.load(stats_path)
            mean = data["mean"]
            std = data["std"]
        else:
            # Fallback to old format
            old_path = PROCESSED_DIR / "norm_stats.npy"
            if not old_path.exists():
                raise FileNotFoundError(
                    f"Normalization stats not found at {stats_path} or {old_path}\n"
                    "Run preprocessing script first."
                )
            
            data = np.load(old_path, allow_pickle=True).item()
            mean = data["mean"]
            std = data["std"]
        
        _norm_stats_cache = (mean, std)
    
    return _norm_stats_cache


def decode_label(label_int: int) -> str:
    """
    Convert integer label to exercise name.
    
    Args:
        label_int: Integer label
        
    Returns:
        Exercise name
    """
    label_map = get_label_map()
    
    if not 0 <= label_int < len(label_map):
        raise ValueError(
            f"Invalid label {label_int}. Valid range: 0-{len(label_map)-1}"
        )
    
    return str(label_map[label_int])


def encode_label(class_name: str) -> int:
    """
    Convert exercise name to integer label.
    
    Args:
        class_name: Exercise name
        
    Returns:
        Integer label
    """
    label_to_int = get_label_to_int()
    
    if class_name not in label_to_int:
        valid_names = ", ".join(label_to_int.keys())
        raise ValueError(
            f"Unknown class name '{class_name}'. "
            f"Valid names: {valid_names}"
        )
    
    return label_to_int[class_name]


def find_split_files(split_name: str) -> List[Path]:
    """
    Find all .npz files for a given split.
    
    Args:
        split_name: One of 'train', 'eval', 'test'
        
    Returns:
        List of paths to .npz files
    """
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {PROCESSED_DIR}\n"
            "Run preprocessing script first."
        )
    
    npz_files = []
    
    # Iterate through exercise directories
    for ex_dir in PROCESSED_DIR.iterdir():
        # Skip non-directories and metadata files
        if not ex_dir.is_dir():
            continue
        
        # Look for split subdirectory
        split_dir = ex_dir / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        
        # Collect all .npz files in this split
        for npz_path in split_dir.glob("*.npz"):
            npz_files.append(npz_path)
    
    return sorted(npz_files)


def load_split(
    split_name: str,
    shuffle: bool = True,
    use_cache: bool = True,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a data split (train/eval/test).
    
    Args:
        split_name: One of 'train', 'eval', 'test'
        shuffle: Whether to shuffle the data (recommended for training)
        use_cache: Whether to use cached data if available
        seed: Random seed for shuffling (None for no seed)
        
    Returns:
        Tuple of (features, labels):
            features: shape (n_samples, sequence_length, n_features)
            labels: shape (n_samples,)
    """
    # Validate split name
    valid_splits = {"train", "eval", "test"}
    if split_name not in valid_splits:
        raise ValueError(
            f"Invalid split_name '{split_name}'. "
            f"Must be one of: {', '.join(valid_splits)}"
        )
    
    # Check cache
    cache_key = (split_name, shuffle, seed)
    if use_cache and cache_key in _data_cache:
        print(f"✅ Loaded {split_name} from cache")
        return _data_cache[cache_key]
    
    # Find all files for this split
    npz_files = find_split_files(split_name)
    
    if not npz_files:
        raise RuntimeError(
            f"No data found for '{split_name}' split in {PROCESSED_DIR}\n"
            f"Expected structure: {PROCESSED_DIR}/<exercise>/{split_name}/*.npz"
        )
    
    # Load all data
    all_X = []
    all_y = []
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            X = data["features"]
            y = data["labels"]
            
            if X.size > 0 and y.size > 0:
                if len(X) != len(y):
                    warnings.warn(
                        f"Shape mismatch in {npz_path.name}: "
                        f"X={X.shape}, y={y.shape}"
                    )
                    continue
                
                all_X.append(X)
                all_y.append(y)
            
        except Exception as e:
            warnings.warn(f"Error loading {npz_path.name}: {e}")
            continue
    
    if not all_X:
        raise RuntimeError(f"No valid data loaded for '{split_name}' split")
    
    # Concatenate all data
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    
    # Shuffle if requested
    if shuffle:
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]
    
    # Validate labels
    n_classes = len(get_label_map())
    unique_labels = np.unique(y)
    
    if len(unique_labels) == 0:
        raise RuntimeError(f"No labels found in '{split_name}' split")
    
    if np.any(unique_labels >= n_classes) or np.any(unique_labels < 0):
        warnings.warn(
            f"Found labels outside valid range [0, {n_classes-1}]: {unique_labels}"
        )
    
    # Cache the result
    if use_cache:
        _data_cache[cache_key] = (X, y)
    
    # Print summary
    print(f"✅ Loaded {split_name}: {X.shape[0]:,} samples, "
          f"{len(unique_labels)} classes, shape={X.shape}")
    
    return X, y


def get_class_distribution(y: np.ndarray) -> Dict[str, int]:
    """
    Get the distribution of samples per class.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary mapping class names to sample counts
    """
    unique, counts = np.unique(y, return_counts=True)
    label_map = get_label_map()
    
    distribution = {}
    for label_int, count in zip(unique, counts):
        if 0 <= label_int < len(label_map):
            class_name = str(label_map[label_int])
            distribution[class_name] = int(count)
    
    return distribution


def print_dataset_info():
    """Print comprehensive information about the dataset."""
    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)
    
    try:
        # Load label map
        label_map = get_label_map()
        print(f"\n📋 Classes ({len(label_map)}):")
        for idx, name in enumerate(label_map):
            print(f"  {idx}: {name}")
        
        # Load normalization stats
        try:
            mean, std = get_normalization_stats()
            print(f"\n📊 Normalization Stats:")
            print(f"  Mean: min={mean.min():.4f}, max={mean.max():.4f}, avg={mean.mean():.4f}")
            print(f"  Std:  min={std.min():.4f}, max={std.max():.4f}, avg={std.mean():.4f}")
        except FileNotFoundError:
            print("\n⚠️  Normalization stats not found")
        
        # Load each split
        print(f"\n📦 Data Splits:")
        for split_name in ["train", "eval", "test"]:
            try:
                X, y = load_split(split_name, shuffle=False)
                dist = get_class_distribution(y)
                
                print(f"\n  {split_name.upper()}:")
                print(f"    Total samples: {len(X):,}")
                print(f"    Shape: {X.shape}")
                print(f"    Distribution:")
                for class_name, count in sorted(dist.items()):
                    percentage = 100 * count / len(y)
                    print(f"      {class_name:15s}: {count:5,} ({percentage:5.1f}%)")
                
            except Exception as e:
                print(f"\n  {split_name.upper()}: ❌ {e}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 70)


def clear_cache():
    """Clear all cached data to free memory."""
    global _data_cache
    _data_cache.clear()
    print("✅ Cache cleared")


# Example usage
if __name__ == "__main__":
    print_dataset_info()
    
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)
    
    # Load training data
    X_train, y_train = load_split("train", shuffle=True)
    
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Label shape: {y_train.shape}")
    print(f"Feature dtype: {X_train.dtype}")
    print(f"Label dtype: {y_train.dtype}")
    
    print(f"\nUnique labels: {np.unique(y_train)}")
    print(f"\nFirst 5 labels: {y_train[:5]}")
    print(f"Decoded: {[decode_label(int(l)) for l in y_train[:5]]}")
    
    # Show class distribution
    print("\n" + "=" * 70)
    print("Class Distribution")
    print("=" * 70)
    dist = get_class_distribution(y_train)
    for class_name, count in sorted(dist.items()):
        print(f"  {class_name:15s}: {count:5,} samples")