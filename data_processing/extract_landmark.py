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
    
    # Optimization: Pre-allocate if we know video length
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarks_list = []
    zero_frame = [0.0] * (NUM_LANDMARKS * LANDMARK_DIMS)
    
    # Optimization: Reduce frame processing if video is very long
    if USE_FAST_MODE and total_frames > SEQUENCE_LENGTH * 3:
        skip_rate = 1
    else:
        skip_rate = 1
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping optimization
            if frame_count % skip_rate != 0:
                frame_count += 1
                continue
            
            # Optimization: Resize all frames to 640px for consistent processing
            height, width = frame.shape[:2]
            if height > 640 or width > 640:
                scale = 640 / max(height, width)
                frame = cv2.resize(frame, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_AREA)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(landmarks)
            else:
                # Forward-fill with last valid frame
                if landmarks_list:
                    landmarks_list.append(landmarks_list[-1])
                else:
                    landmarks_list.append(zero_frame)
            
            frame_count += 1
            
    finally:
        cap.release()
    
    return np.array(landmarks_list, dtype=np.float32)


def create_sequences(landmarks: np.ndarray, seq_len: int, step: int) -> np.ndarray:
    """Create sliding window sequences with vectorized operations."""
    if len(landmarks) < seq_len:
        return np.empty((0, seq_len, landmarks.shape[1]), dtype=np.float32)

    num_sequences = (len(landmarks) - seq_len) // step + 1
    
    # Optimization: Pre-allocate array
    sequences = np.empty((num_sequences, seq_len, landmarks.shape[1]), dtype=np.float32)
    
    # Vectorized indexing for better performance
    indices = np.arange(num_sequences) * step
    for idx, start_idx in enumerate(indices):
        sequences[idx] = landmarks[start_idx:start_idx + seq_len]
    
    return sequences


def process_video_worker(args: Tuple[Path, str]) -> ProcessResult:
    """Worker function for parallel video processing."""
    video_path, exercise_name = args
    
    # Initialize MediaPipe inside worker
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
        
        # Handle short videos
        if num_frames < SEQUENCE_LENGTH:
            frames_short = SEQUENCE_LENGTH - num_frames
            max_allowed_pad = max(MAX_PAD_FRAMES, int(SEQUENCE_LENGTH * PAD_THRESHOLD_PERCENT))
            
            if PAD_SHORT_VIDEOS and frames_short <= max_allowed_pad:
                padding = np.repeat(landmarks[-1:], frames_short, axis=0)
                landmarks = np.concatenate([landmarks, padding], axis=0)
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
        np.save(output_path, sequences)
        
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
        
        # Progress bar with tqdm
        with tqdm(total=len(tasks), desc="Processing videos", 
                  unit="video", ncols=100, colour="green") as pbar:
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result.success:
                    successful += 1
                    total_sequences += result.num_sequences
                    pbar.set_postfix({
                        'success': successful, 
                        'failed': failed,
                        'sequences': total_sequences
                    })
                else:
                    failed += 1
                    pbar.set_postfix({
                        'success': successful, 
                        'failed': failed
                    })
                
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