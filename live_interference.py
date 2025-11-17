# live_exercise_app.py (smoothed prediction version)
import streamlit as st
import torch
import numpy as np
import cv2
from collections import deque
from pathlib import Path

import mediapipe as mp

from models.autoencoder import FrameAutoencoder
from models.lstm import ExerciseClassifier
from dataset.dataset_loader import get_label_map
import config

CONF_THRESHOLD = 0.6 

# --- Paths ---
NORM_STATS_PATH = Path("data/processed/normalization_stats.npz")

# --- Device ---
DEVICE = config.DEVICE

# --- Load normalization stats ---
norm_data = np.load(NORM_STATS_PATH)
mean = norm_data["mean"]
std = norm_data["std"]

# --- Load label map ---
label_map = get_label_map()
NUM_CLASSES = len(label_map)
class_names = list(label_map) if not isinstance(label_map, dict) else list(label_map.keys())

# --- Load Autoencoder ---
autoencoder = FrameAutoencoder(
    input_dim=config.INPUT_DIM,
    latent_dim=config.AE_LATENT_DIM,
    dropout=config.AE_DROPOUT
).to(DEVICE)
autoencoder.eval()

# --- Load Classifier ---
model_saved = torch.load(config.LSTM_BEST, map_location=DEVICE, weights_only=False)
model = ExerciseClassifier(
    autoencoder=autoencoder,
    num_classes=NUM_CLASSES,
    input_dim=config.INPUT_DIM if not config.USE_AUTOENCODER else config.AE_LATENT_DIM,
    hidden_dim=config.LSTM_HIDDEN_DIM,
    freeze_encoder=config.FREEZE_ENCODER,
    use_autoencoder=config.USE_AUTOENCODER,
    use_bilstm=config.LSTM_BIDIRECTIONAL,
    dropout=config.LSTM_DROPOUT
).to(DEVICE)
model.load_state_dict(model_saved['model_state_dict'])
model.eval()

# --- Streamlit UI ---
st.title("Live Exercise Classifier with Pose")
st.write("Webcam feed with landmarks and smooth exercise predictions.")

run = st.checkbox("Start Camera")
camera_placeholder = st.empty()
prediction_placeholder = st.empty()

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# --- Frame buffer for LSTM ---
BUFFER_SIZE = config.EXPECTED_SEQUENCE_LENGTH
frame_buffer = deque(maxlen=BUFFER_SIZE)
prob_buffer = deque(maxlen=10)  # store last 10 softmax outputs for smoothing

EMA_ALPHA = 0.3
ema_probs = None

# --- Camera capture ---
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("No camera input detected")
        break

    # --- Flip horizontally ---
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Pose landmarks ---
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        # --- Extract landmarks ---
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        landmarks = (landmarks - mean) / std
        frame_buffer.append(landmarks)

        # --- Predict when buffer full ---
        if len(frame_buffer) == BUFFER_SIZE:
            seq_input = np.array(frame_buffer)[None, :, :]  # (1, seq_len, input_dim)
            seq_tensor = torch.from_numpy(seq_input).float().to(DEVICE)

            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # --- Add to prob buffer ---
            prob_buffer.append(probs)

            # --- Smooth using moving average ---
            avg_probs = np.mean(np.array(prob_buffer), axis=0)

            # --- Apply EMA ---
            if ema_probs is None:
                ema_probs = avg_probs
            else:
                ema_probs = EMA_ALPHA * avg_probs + (1 - EMA_ALPHA) * ema_probs

            pred_idx = np.argmax(ema_probs)
            confidence = ema_probs[pred_idx]
            if confidence < CONF_THRESHOLD:
                pred_class = "Unknown"
            else:
                pred_class = class_names[pred_idx]

            # --- Show prediction ---
            prediction_placeholder.markdown(
                f"**Prediction:** {pred_class}  \n**Confidence:** {confidence*100:.1f}%"
            )

    camera_placeholder.image(frame_rgb, channels="RGB")

cap.release()
st.write("Camera stopped.")
