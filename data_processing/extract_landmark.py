import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import config

# Suppress MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GLOG_minloglevel'] = '3'  # Suppress glog from MediaPipe

# Configuration
RAW_VIDEO_DIR = Path("data/raw_videos")
LANDMARK_DIR = Path("data/landmarks")
SEQUENCE_LENGTH = config.EXPECTED_SEQUENCE_LENGTH
STEP = config.STEP
NUM_JOBS = max(1, (os.cpu_count() or 2) // 2)
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3

# Video handling
PAD_SHORT_VIDEOS = True
MAX_PAD_FRAMES = 3
PAD_THRESHOLD_PERCENT = 0.05

# Optimization settings
BATCH_READ_FRAMES = 10  # Read frames in batches for better I/O
USE_FAST_MODE = True    # Skip quality checks for speed


@dataclass
class ProcessResult:
    """Track processing results."""
    video_name: str
    success: bool
    num_sequences: int = 0
    error_msg: Optional[str] = None


def extract_landmarks_from_video(video_path: Path, pose_detector) -> np.ndarray:
    """Extract pose landmarks from video with optimized frame reading."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([], dtype=np.float32)
    
    # Use list for dynamic allocation - no frame limits
    landmarks_list = []
    
    # Pre-create zero frame as numpy array for efficiency
    zero_frame = np.zeros(NUM_LANDMARKS * LANDMARK_DIMS, dtype=np.float32)
    last_valid_landmarks = None
    
    # Pre-calculate resize parameters if needed
    first_frame_read = False
    target_size = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # One-time resize calculation
            if not first_frame_read:
                height, width = frame.shape[:2]
                if height > 640 or width > 640:
                    scale = 640 / max(height, width)
                    target_size = (int(width * scale), int(height * scale))
                first_frame_read = True
            
            # Apply resize if needed (using pre-calculated size)
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            
            # Process frame - direct RGB conversion without intermediate
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                # Optimized: Direct numpy extraction without list comprehension
                landmarks = np.empty(NUM_LANDMARKS * LANDMARK_DIMS, dtype=np.float32)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    idx = i * 3
                    landmarks[idx] = lm.x
                    landmarks[idx + 1] = lm.y
                    landmarks[idx + 2] = lm.z
                
                last_valid_landmarks = landmarks
            else:
                # Forward-fill with last valid frame
                landmarks = last_valid_landmarks if last_valid_landmarks is not None else zero_frame
            
            # Store landmark - no limits, captures everything
            landmarks_list.append(landmarks)
            
    finally:
        cap.release()
    
    # Convert to numpy array efficiently
    if not landmarks_list:
        return np.array([], dtype=np.float32)
    
    return np.vstack(landmarks_list)


def create_sequences(landmarks: np.ndarray, seq_len: int, step: int) -> np.ndarray:
    """Create sliding window sequences using stride tricks for maximum performance."""
    if len(landmarks) < seq_len:
        return np.empty((0, seq_len, landmarks.shape[1]), dtype=np.float32)

    num_sequences = (len(landmarks) - seq_len) // step + 1
    
    # Ultra-optimized: Use stride tricks to create view without copying
    shape = (num_sequences, seq_len, landmarks.shape[1])
    strides = (landmarks.strides[0] * step, landmarks.strides[0], landmarks.strides[1])
    
    sequences = np.lib.stride_tricks.as_strided(
        landmarks, 
        shape=shape, 
        strides=strides,
        writeable=False
    )
    
    # Copy only when saving (done in worker), not here - saves memory and time
    return sequences.astype(np.float32, copy=True)


def process_video_worker(args: Tuple[Path, str]) -> ProcessResult:
    """Worker function for parallel video processing."""
    video_path, exercise_name = args
    
    # Initialize MediaPipe inside worker with optimized settings
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Full quality model - captures all details
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,  # Enable smoothing for better temporal consistency
        enable_segmentation=False  # Disable segmentation for speed
    )
    
    try:
        landmarks = extract_landmarks_from_video(video_path, pose_detector)
        
        if landmarks.size == 0:
            return ProcessResult(
                video_name=video_path.name,
                success=False,
                error_msg="Video corrupted or empty"
            )

        num_frames = len(landmarks)
        
        # Pre-calculate max allowed padding (moved outside if statement)
        max_allowed_pad = max(MAX_PAD_FRAMES, int(SEQUENCE_LENGTH * PAD_THRESHOLD_PERCENT))
        
        # Handle short videos
        if num_frames < SEQUENCE_LENGTH:
            frames_short = SEQUENCE_LENGTH - num_frames
            
            if PAD_SHORT_VIDEOS and frames_short <= max_allowed_pad:
                # Optimized padding using tile
                padding = np.tile(landmarks[-1], (frames_short, 1))
                landmarks = np.vstack([landmarks, padding])
            else:
                return ProcessResult(
                    video_name=video_path.name,
                    success=False,
                    error_msg=f"Too short: {num_frames}/{SEQUENCE_LENGTH} frames"
                )
        
        sequences = create_sequences(landmarks, SEQUENCE_LENGTH, STEP)
        
        if sequences.shape[0] == 0:
            return ProcessResult(
                video_name=video_path.name,
                success=False,
                error_msg="No sequences created"
            )
        
        # Save sequences
        output_dir = LANDMARK_DIR / exercise_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{video_path.stem}.npy"
        np.save(output_path, sequences)  # Already float32 from create_sequences
        
        return ProcessResult(
            video_name=video_path.name,
            success=True,
            num_sequences=sequences.shape[0]
        )
        
    except Exception as e:
        return ProcessResult(
            video_name=video_path.name,
            success=False,
            error_msg=str(e)
        )
    finally:
        pose_detector.close()


def collect_video_tasks() -> List[Tuple[Path, str]]:
    """Collect all video processing tasks."""
    tasks = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    if not RAW_VIDEO_DIR.exists():
        raise FileNotFoundError(f"Raw video directory not found: {RAW_VIDEO_DIR}")
    
    for exercise_dir in RAW_VIDEO_DIR.iterdir():
        if not exercise_dir.is_dir():
            continue
        
        exercise_name = exercise_dir.name
        video_files = [
            f for f in exercise_dir.iterdir() 
            if f.suffix.lower() in video_extensions
        ]
        
        if video_files:
            print(f"📁 Found {len(video_files)} videos for: {exercise_name}")
            tasks.extend((video_path, exercise_name) for video_path in video_files)
    
    return tasks


def main():
    """Main processing pipeline with progress tracking."""
    print("=" * 70)
    print("🏋️  LANDMARK EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"⚙️  Workers: {NUM_JOBS}")
    print(f"⚙️  Sequence length: {SEQUENCE_LENGTH} | Step: {STEP}")
    
    if PAD_SHORT_VIDEOS:
        max_pad = max(MAX_PAD_FRAMES, int(SEQUENCE_LENGTH * PAD_THRESHOLD_PERCENT))
        print(f"⚙️  Padding: Enabled (max {max_pad} frames)")
    else:
        print(f"⚙️  Padding: Disabled (min {SEQUENCE_LENGTH} frames required)")
    
    print("=" * 70)
    
    # Collect tasks
    print("\n📂 Scanning directories...")
    tasks = collect_video_tasks()
    
    if not tasks:
        print("⚠️  No videos found to process!")
        return
    
    print(f"\n✅ Found {len(tasks)} videos to process\n")
    
    # Create output directory
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process with progress bar
    results = []
    successful = 0
    failed = 0
    total_sequences = 0
    
    with ProcessPoolExecutor(max_workers=NUM_JOBS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_video_worker, task): task for task in tasks}
        
        # Simple progress bar without postfix updates
        with tqdm(total=len(tasks), desc="Processing videos", 
                  unit="video", ncols=70, colour="green") as pbar:
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result.success:
                    successful += 1
                    total_sequences += result.num_sequences
                else:
                    failed += 1
                
                pbar.update(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {successful}/{len(tasks)} ({successful/len(tasks)*100:.1f}%)")
    print(f"❌ Failed: {failed}/{len(tasks)} ({failed/len(tasks)*100:.1f}%)")
    print(f"📦 Total sequences created: {total_sequences:,}")
    
    # Show failed videos if any
    if failed > 0:
        print("\n❌ Failed videos:")
        for result in results:
            if not result.success:
                print(f"   • {result.video_name}: {result.error_msg}")
    
    print("=" * 70)
    print("✨ Pipeline complete!\n")


if __name__ == "__main__":
    main()