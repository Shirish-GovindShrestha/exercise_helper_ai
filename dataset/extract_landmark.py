import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List

# Configuration
RAW_VIDEO_DIR = Path("data/raw_videos")
LANDMARK_DIR = Path("data/landmarks")
SEQUENCE_LENGTH = 90
STEP = 1
# Optimization: Prevent CPU thrashing. MediaPipe is heavy.
NUM_JOBS = max(1, (os.cpu_count() or 2) // 2) 
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3

# Video length handling
PAD_SHORT_VIDEOS = True
MAX_PAD_FRAMES = 3
PAD_THRESHOLD_PERCENT = 0.05

def extract_landmarks_from_video(video_path: Path, pose_detector) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # Log error instead of raising to prevent crashing the whole pool
        print(f"❌ Error: Cannot open {video_path}")
        return np.array([], dtype=np.float32)
    
    landmarks_list = []
    # Initialize with zeros, but consider skipping logic later
    zero_frame = [0.0] * (NUM_LANDMARKS * LANDMARK_DIMS)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(landmarks)
            else:
                # DATA IMPROVEMENT: Forward-fill instead of zero-fill if possible
                if landmarks_list:
                    landmarks_list.append(landmarks_list[-1])
                else:
                    landmarks_list.append(zero_frame)
    finally:
        cap.release()
    
    return np.array(landmarks_list, dtype=np.float32)

def create_sequences(landmarks: np.ndarray, seq_len: int, step: int) -> np.ndarray:
    """Create sliding window sequences from landmark data."""
    # SAFETY CHECK: Prevent negative dimensions
    if len(landmarks) < seq_len:
        return np.empty((0, seq_len, landmarks.shape[1]), dtype=np.float32)

    num_sequences = (len(landmarks) - seq_len) // step + 1
    
    sequences = np.empty((num_sequences, seq_len, landmarks.shape[1]), dtype=np.float32)
    
    for idx, i in enumerate(range(0, len(landmarks) - seq_len + 1, step)):
        sequences[idx] = landmarks[i:i + seq_len]
    
    return sequences

def process_video_worker(args: Tuple[Path, str]) -> None:
    video_path, exercise_name = args
    
    # Init MediaPipe inside worker
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        landmarks = extract_landmarks_from_video(video_path, pose_detector)
        
        # SAFETY CHECK: Handle empty videos
        if landmarks.size == 0:
            print(f"⚠️  Skipping {video_path.name} - Video corrupted or empty")
            return

        num_frames = len(landmarks)
        
        if num_frames < SEQUENCE_LENGTH:
            frames_short = SEQUENCE_LENGTH - num_frames
            shortage_percent = (frames_short / SEQUENCE_LENGTH) * 100
            max_allowed_pad = max(MAX_PAD_FRAMES, int(SEQUENCE_LENGTH * PAD_THRESHOLD_PERCENT))
            
            if PAD_SHORT_VIDEOS and frames_short <= max_allowed_pad:
                # Pad logic
                padding = np.repeat(landmarks[-1:], frames_short, axis=0)
                landmarks = np.concatenate([landmarks, padding], axis=0)
                print(f"  ✅ Padded {video_path.name}")
            else:
                # Video too short
                return 
        
        sequences = create_sequences(landmarks, SEQUENCE_LENGTH, STEP)
        
        if sequences.shape[0] == 0:
            return
        
        output_dir = LANDMARK_DIR / exercise_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{video_path.stem}.npy"
        np.save(output_path, sequences)
        
    except Exception as e:
        print(f"❌ Error processing {video_path.name}: {e}")
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
            print(f"Found {len(video_files)} videos for exercise: {exercise_name}")
            for video_path in video_files:
                tasks.append((video_path, exercise_name))
    
    return tasks


def main():
    """Main processing pipeline."""
    print(f"Starting landmark extraction with {NUM_JOBS} workers...")
    print(f"Sequence length: {SEQUENCE_LENGTH}, Step: {STEP}")
    
    if PAD_SHORT_VIDEOS:
        max_pad = max(MAX_PAD_FRAMES, int(SEQUENCE_LENGTH * PAD_THRESHOLD_PERCENT))
        print(f"Padding enabled: Videos within {max_pad} frames of target will be padded")
    else:
        print(f"Padding disabled: Only videos with ≥{SEQUENCE_LENGTH} frames will be processed")
    
    # Collect all tasks
    tasks = collect_video_tasks()
    
    if not tasks:
        print("⚠️  No videos found to process!")
        return
    
    print(f"\nProcessing {len(tasks)} videos...")
    
    # Create output directory
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process videos in parallel
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=NUM_JOBS) as executor:
        futures = {executor.submit(process_video_worker, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                future.result()
                successful += 1
            except Exception as e:
                failed += 1
                task = futures[future]
                print(f"❌ Process failed for {task[0].name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()