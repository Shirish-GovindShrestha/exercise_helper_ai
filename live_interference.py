import streamlit as st
import torch
import numpy as np
import cv2
from collections import deque, Counter
import mediapipe as mp

# --- Local Modules ---
from models.lstm import ExerciseClassifier
import config
from data_processing.dataset_loader import get_label_map

# --- Constants ---
DEVICE = config.DEVICE
BUFFER_SIZE = 30  # reduced from 60 for faster response
EMA_ALPHA = 0.8  # increased for faster adaptation
HISTORY_LEN = 3  # reduced from 5 for faster voting
CONF_THRESHOLD = 0.55  # slightly lower for faster detection
SKIP_FRAMES = 2  # process every nth frame

# Feature mode from config
INPUT_FEATURE = config.INPUT_FEATURE  # "landmarks", "angles", or "combined"

@st.cache_resource
def load_resources():
    # Load Labels
    label_map = get_label_map()
    class_names = list(label_map) if not isinstance(label_map, dict) else list(label_map.keys())
    num_classes = len(class_names)
    
    return num_classes, class_names

NUM_CLASSES, CLASS_NAMES = load_resources()

# --- Feature Dimensions ---
LANDMARKS_DIM = 99  # 33 landmarks × 3
ANGLES_DIM = 12

if INPUT_FEATURE == "landmarks":
    INPUT_DIM = LANDMARKS_DIM
elif INPUT_FEATURE == "angles":
    INPUT_DIM = ANGLES_DIM
elif INPUT_FEATURE == "combined":
    INPUT_DIM = ANGLES_DIM + LANDMARKS_DIM
else:
    raise ValueError(f"Invalid INPUT_FEATURE: {INPUT_FEATURE}")

# --- Angles Definition (12 angles) ---
JOINT_CHAINS = [
    # --- ARMS (4 angles) ---
    [12, 14, 16],  # Right shoulder
    [11, 13, 15],  # Left shoulder
    [14, 12, 16],  # Right elbow
    [13, 11, 15],  # Left elbow

    # --- LEGS (6 angles) ---
    [24, 23, 26],  # Right hip
    [23, 24, 25],  # Left hip
    [26, 24, 28],  # Right knee
    [25, 23, 27],  # Left knee
    [28, 26, 32],  # Right ankle
    [27, 25, 31],  # Left ankle

    # --- TORSO (2 angles) ---
    [12, 24, 26],  # Torso inclination
    [11, 23, 12],  # Spine alignment
]

# --- Helper Functions ---
def vector_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    ba_n = ba / (np.linalg.norm(ba) + 1e-6)
    bc_n = bc / (np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(np.dot(ba_n, bc_n), -1.0, 1.0)
    return np.degrees(np.arccos(cosine)) / 180.0  # normalize 0-1

def landmarks_to_angles(landmarks):
    """Convert 33 landmarks (Nx3) -> 12 angles."""
    angles = []
    for a, mid, c in JOINT_CHAINS:
        A, B, C = landmarks[a], landmarks[mid], landmarks[c]
        angles.append(vector_angle(A, B, C))
    return np.array(angles, dtype=np.float32)

def extract_features(landmarks):
    """
    Extract features based on INPUT_FEATURE mode.
    
    Args:
        landmarks: (33, 3) array of pose landmarks
        
    Returns:
        Feature array based on mode:
        - "landmarks": (99,) flattened landmarks
        - "angles": (12,) joint angles
        - "combined": (111,) angles + landmarks
    """
    if INPUT_FEATURE == "landmarks":
        return landmarks.flatten()  # (99,)
    
    elif INPUT_FEATURE == "angles":
        return landmarks_to_angles(landmarks)  # (12,)
    
    elif INPUT_FEATURE == "combined":
        angles = landmarks_to_angles(landmarks)  # (12,)
        landmarks_flat = landmarks.flatten()  # (99,)
        return np.concatenate([angles, landmarks_flat])  # (111,)
    
    else:
        raise ValueError(f"Invalid INPUT_FEATURE: {INPUT_FEATURE}")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = ExerciseClassifier(
        autoencoder=None,
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        freeze_encoder=False,
        use_autoencoder=False,
        use_bilstm=config.LSTM_BIDIRECTIONAL,
        dropout=config.LSTM_DROPOUT
    ).to(DEVICE)
    checkpoint = torch.load(config.LSTM_BEST, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# --- Streamlit UI ---
st.title(f"Live Exercise Classifier ({INPUT_FEATURE.title()})")
st.caption(f"Using {INPUT_DIM} features: {INPUT_FEATURE}")

run = st.checkbox("Start Camera", value=False)
camera_placeholder = st.empty()
prediction_placeholder = st.empty()
stats_placeholder = st.empty()

# --- Mediapipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # reduced from 1 for faster inference
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- State ---
frame_buffer = deque(maxlen=BUFFER_SIZE)
prediction_history = deque(maxlen=HISTORY_LEN)
ema_probs = None
frame_count = 0  # for frame skipping

# --- Main Loop ---
if run:
    cap = cv2.VideoCapture(0)
    
    # Warmup camera
    for _ in range(5):
        cap.read()
    
    try:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("No camera detected.")
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % SKIP_FRAMES != 0:
                camera_placeholder.image(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB), 
                                        channels="RGB", use_container_width=True)
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Extract landmarks (33, 3)
                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in results.pose_landmarks.landmark
                ], dtype=np.float32)
                
                # Extract features based on mode
                features = extract_features(landmarks)
                frame_buffer.append(features)

                # Predict when buffer is full
                if len(frame_buffer) == BUFFER_SIZE:
                    seq_input = np.array(frame_buffer)[None, :, :]  # (1, seq_len, input_dim)
                    seq_tensor = torch.from_numpy(seq_input).float().to(DEVICE)
                    
                    with torch.no_grad():
                        logits = model(seq_tensor)
                        current_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    # EMA smoothing
                    if ema_probs is None:
                        ema_probs = current_probs
                    else:
                        ema_probs = EMA_ALPHA * current_probs + (1 - EMA_ALPHA) * ema_probs

                    # Voting mechanism (faster with reduced history)
                    candidate_idx = np.argmax(ema_probs)
                    prediction_history.append(candidate_idx)
                    most_common_idx, freq = Counter(prediction_history).most_common(1)[0]
                    final_idx = most_common_idx if freq >= 2 else candidate_idx  # reduced threshold
                    confidence = ema_probs[final_idx]

                    # Display prediction
                    if confidence < CONF_THRESHOLD:
                        prediction_placeholder.markdown("### Status: Unknown / Idle")
                        stats_placeholder.text(f"Max confidence: {confidence*100:.1f}%")
                    else:
                        pred_name = CLASS_NAMES[final_idx]
                        prediction_placeholder.markdown(
                            f"### Prediction: **{pred_name}** ({confidence*100:.1f}%)"
                        )
                        
                        # Show top 3 predictions
                        top3_idx = np.argsort(ema_probs)[-3:][::-1]
                        stats_text = "Top 3:\n"
                        for idx in top3_idx:
                            stats_text += f"  {CLASS_NAMES[idx]}: {ema_probs[idx]*100:.1f}%\n"
                        stats_placeholder.text(stats_text)

            else:
                frame_buffer.clear()
                ema_probs = None
                prediction_placeholder.markdown("### Status: No Person Detected")
                stats_placeholder.text("")
            
            # Display camera feed
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    finally:
        cap.release()
        st.write("Camera stopped.")
else:
    st.info("👆 Check the box above to start the camera")