import streamlit as st
import torch
import numpy as np
import cv2
from collections import deque, Counter
from pathlib import Path

import mediapipe as mp

# --- Local Modules ---
from models.autoencoder import FrameAutoencoder
from models.lstm import ExerciseClassifier
from dataset.dataset_loader import get_label_map
import config

# --- Constants & Config ---
CONF_THRESHOLD = 0.6 
NORM_STATS_PATH = Path("data/processed/normalization_stats.npz")
DEVICE = config.DEVICE

# --- NEW: Landmark Constants for Centering ---
# MediaPipe landmark indices (0-32)
LH_IDX, RH_IDX = 23, 24 # Left Hip, Right Hip
LS_IDX, RS_IDX = 11, 12 # Left Shoulder, Right Shoulder
LANDMARK_DIMS = 3 # x, y, z

# --- Prediction Parameters ---
BUFFER_SIZE = config.EXPECTED_SEQUENCE_LENGTH
EMA_ALPHA = 0.7      # High alpha = faster visual reaction
HISTORY_LEN = 5      # "Voting" window size (prevents flickering)


# --- 1. Load Data & Models (Cached) ---

@st.cache_resource
def load_resources():
    """Loads normalization stats and label map once."""
    # Load Stats
    norm_data = np.load(NORM_STATS_PATH)
    mean = norm_data["mean"].astype(np.float32)
    std = norm_data["std"].astype(np.float32)
    
    # Load Labels
    label_map = get_label_map()
    class_names = list(label_map) if not isinstance(label_map, dict) else list(label_map.keys())
    num_classes = len(class_names)
    
    return mean, std, class_names, num_classes


@st.cache_resource
def load_models(num_classes):
    """Loads PyTorch models once and keeps them in memory."""
    # Load Autoencoder
    ae = FrameAutoencoder(
        input_dim=config.INPUT_DIM,
        latent_dim=config.AE_LATENT_DIM,
        dropout=config.AE_DROPOUT
    ).to(DEVICE)
    ae.eval()

    # Load Classifier
    lstm_model = ExerciseClassifier(
        autoencoder=ae,
        num_classes=num_classes,
        input_dim=config.INPUT_DIM if not config.USE_AUTOENCODER else config.AE_LATENT_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        freeze_encoder=config.FREEZE_ENCODER,
        use_autoencoder=config.USE_AUTOENCODER,
        use_bilstm=config.LSTM_BIDIRECTIONAL,
        dropout=config.LSTM_DROPOUT
    ).to(DEVICE)
    
    # Load Weights
    checkpoint = torch.load(config.LSTM_BEST, map_location=DEVICE, weights_only=False)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    return ae, lstm_model

# --- Initialize Resources ---
mean, std, class_names, NUM_CLASSES = load_resources()
autoencoder, model = load_models(NUM_CLASSES)

# --- Streamlit UI ---
st.title("Live Exercise Classifier")
st.markdown("Webcam feed with **Instant Reaction** & **Anti-Flicker** technology.")

run = st.checkbox("Start Camera", value=False)

# Layout containers
col1, col2 = st.columns([3, 2])
with col1:
    camera_placeholder = st.empty()
with col2:
    prediction_placeholder = st.empty()
    stats_placeholder = st.empty()

# --- Mediapipe Pose Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# --- State Variables ---
frame_buffer = deque(maxlen=BUFFER_SIZE)
prediction_history = deque(maxlen=HISTORY_LEN) # Stores last N class indices for voting
ema_probs = None  # For visual smoothing of confidence numbers

# --- Main Loop ---
cap = cv2.VideoCapture(0)

# Warmup camera feed to discard stale frames (helps prevent 98% stall)
for _ in range(5):
    cap.read() 

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("No camera input detected.")
        break

    # Flip & Convert
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose Processing
    results = pose.process(frame_rgb)
    
    current_status = "Waiting for person..."
    
    if results.pose_landmarks:
        # Draw Skeleton
        mp_drawing.draw_landmarks(
            frame_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        # --- NEW: Extract and Apply Body-Centric Normalization ---
        # 1. Extract raw landmarks (33, 3)
        landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        
        # 2. Find Anchor Point (Translation)
        lh_coords, rh_coords = landmarks_raw[LH_IDX, :], landmarks_raw[RH_IDX, :]
        ls_coords, rs_coords = landmarks_raw[LS_IDX, :], landmarks_raw[RS_IDX, :]
        
        # Use Mid-Hip primary, Mid-Shoulder fallback
        if not np.all(lh_coords == 0) and not np.all(rh_coords == 0):
            center_point = (lh_coords + rh_coords) / 2
        elif not np.all(ls_coords == 0) and not np.all(rs_coords == 0):
            center_point = (ls_coords + rs_coords) / 2
        else:
            center_point = np.zeros(LANDMARK_DIMS, dtype=np.float32) # Fallback to origin
        
        # 3. Translation
        landmarks_centered = landmarks_raw - center_point
        
        # 4. Scaling (Body Size Normalization)
        # Use shoulder distance for live scaling
        current_shoulder_dist = np.linalg.norm(ls_coords - rs_coords)
        scale_divisor = current_shoulder_dist if current_shoulder_dist > 1e-6 else 1.0
        
        landmarks_scaled = landmarks_centered / scale_divisor
        
        # 5. Flatten and Z-Score Standardization
        landmarks_flat = landmarks_scaled.flatten()
        landmarks_final = (landmarks_flat - mean) / std
        
        frame_buffer.append(landmarks_final)
        # --- END: Body-Centric Normalization ---


        # --- Prediction Logic ---
        if len(frame_buffer) == BUFFER_SIZE:
            # Prepare Input
            seq_input = np.array(frame_buffer)[None, :, :]  # (1, seq_len, input_dim)
            seq_tensor = torch.from_numpy(seq_input).float().to(DEVICE)

            with torch.no_grad():
                logits = model(seq_tensor)
                current_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 1. Visual Smoothing (EMA)
            if ema_probs is None:
                ema_probs = current_probs
            else:
                ema_probs = EMA_ALPHA * current_probs + (1 - EMA_ALPHA) * ema_probs

            # 2. Decision Stability (Voting)
            candidate_idx = np.argmax(ema_probs)
            prediction_history.append(candidate_idx)
            
            # Get most common class in recent history
            count = Counter(prediction_history)
            most_common_idx, frequency = count.most_common(1)[0]
            
            # Only switch label if 3 out of 5 frames agree
            if frequency >= 3:
                final_pred_idx = most_common_idx
            else:
                final_pred_idx = candidate_idx 

            confidence = ema_probs[final_pred_idx]

            # 3. Display Logic
            if confidence < CONF_THRESHOLD:
                current_status = "Unknown / Idle"
                color = "gray"
            else:
                pred_name = class_names[final_pred_idx]
                color = "green" if confidence > 0.85 else "orange"
                
                prediction_placeholder.markdown(
                    f"""
                    ### Prediction:
                    # :{color}[{pred_name}]
                    **Confidence:** {confidence*100:.1f}%
                    """
                )
                
                # Optional: Show stats for top 3 classes
                top3_indices = np.argsort(ema_probs)[::-1][:3]
                stats_text = "**Top Probabilities:**\n\n"
                for idx in top3_indices:
                    stats_text += f"- {class_names[idx]}: {ema_probs[idx]*100:.1f}%\n"
                stats_placeholder.markdown(stats_text)

        else:
            # Buffer Filling (Warmup)
            fill_percent = int((len(frame_buffer) / BUFFER_SIZE) * 100)
            prediction_placeholder.markdown(
                f"""
                ### Status: 🟡 Calibrating...
                Gathering motion data: **{fill_percent}%**
                """
            )
            stats_placeholder.empty()

    else:
        # No person detected
        prediction_placeholder.markdown("### Status: 🔴 No Person Detected")
        stats_placeholder.empty()
        # Clear buffer when person leaves to prevent 98% stall
        frame_buffer.clear() 

    camera_placeholder.image(frame_rgb, channels="RGB")

cap.release()
st.write("Camera stopped.")