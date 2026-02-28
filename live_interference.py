import streamlit as st
import torch
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import config
from models.tcn import ExerciseClassifier

# Page config
st.set_page_config(
    page_title="Exercise Classifier",
    page_icon="🏋️",
    layout="wide"
)

# Constants from preprocessing
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3
EXPECTED_LANDMARKS = NUM_LANDMARKS * LANDMARK_DIMS  # 99
ANGLES_DIM = 12
TOTAL_FEATURES = ANGLES_DIM + EXPECTED_LANDMARKS  # 111

# Landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
TORSO_SCALE_EPSILON = 1e-6

# Joint chains for angle calculation
JOINT_CHAINS = [
    # Arms (4 angles)
    [12, 14, 16],  # Right shoulder
    [11, 13, 15],  # Left shoulder
    [14, 12, 16],  # Right elbow
    [13, 11, 15],  # Left elbow
    # Legs (6 angles)
    [24, 23, 26],  # Right hip
    [23, 24, 25],  # Left hip
    [26, 24, 28],  # Right knee
    [25, 23, 27],  # Left knee
    [28, 26, 32],  # Right ankle
    [27, 25, 31],  # Left ankle
    # Torso (2 angles)
    [12, 24, 26],  # Torso inclination
    [11, 23, 12],  # Spine alignment
]

# Initialize MediaPipe
@st.cache_resource
def init_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        checkpoint = torch.load(config.GRU_BEST, map_location=config.DEVICE, weights_only=False)
        
        # Get model config from checkpoint
        num_classes = checkpoint['num_classes']
        label_map = checkpoint['label_map']
        
        # Infer architecture from state_dict
        state_dict = checkpoint['model_state_dict']
        
        # Count GRU layers by checking layer indices in keys
        max_layer = 0
        for key in state_dict.keys():
            if 'gru.weight_ih_l' in key:
                layer_num = int(key.split('_l')[1].split('_')[0])
                max_layer = max(max_layer, layer_num)
        gru_num_layers = max_layer + 1
        
        # Check if bidirectional
        use_bilstm = any('_reverse' in key for key in state_dict.keys())
        
        # Infer hidden_dim from weight shape
        # For GRU: weight_hh_l0 has shape [3*hidden_dim, hidden_dim]
        weight_hh_key = 'gru.weight_hh_l0'
        if weight_hh_key in state_dict:
            hidden_dim = state_dict[weight_hh_key].shape[1]
        else:
            hidden_dim = checkpoint.get('hidden_dim', 96)
        
        # Get dropout (default to checkpoint value or 0.2)
        dropout = checkpoint.get('dropout', 0.2)
        
        # Create classifier with inferred hyperparameters
        model = ExerciseClassifier(
            num_classes=num_classes,
            input_dim=config.INPUT_DIM,
            hidden_dim=hidden_dim,
            lstm_num_layers=gru_num_layers,
            use_bilstm=use_bilstm,
            dropout=dropout
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.DEVICE)
        model.eval()
        
        st.success(f"✅ Loaded model: {gru_num_layers} layers, hidden_dim={hidden_dim}, bidirectional={use_bilstm}")
        
        return model, label_map, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def extract_landmarks(results):
    """Extract pose landmarks as raw feature vector (99 features: x,y,z for 33 landmarks)"""
    if not results.pose_landmarks:
        return None
    
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    return np.array(landmarks, dtype=np.float32)

def relative_normalize_landmarks(landmarks):
    """
    Apply relative normalization: center to mid-hip and scale by torso length
    Input: (99,) array -> Output: (99,) array
    """
    # Reshape to (33, 3)
    lm_reshaped = landmarks.reshape(NUM_LANDMARKS, 3)
    
    # 1. Calculate Mid-Hip (center point)
    P_mid_hip = (lm_reshaped[LEFT_HIP] + lm_reshaped[RIGHT_HIP]) / 2.0
    
    # 2. Center all landmarks to mid-hip
    lm_centered = lm_reshaped - P_mid_hip
    
    # 3. Calculate Mid-Shoulder
    P_mid_shoulder = (lm_centered[LEFT_SHOULDER] + lm_centered[RIGHT_SHOULDER]) / 2.0
    
    # 4. Calculate torso length (scaling factor)
    D_scale = np.linalg.norm(P_mid_shoulder)
    
    # Prevent division by zero
    if D_scale < TORSO_SCALE_EPSILON:
        D_scale = TORSO_SCALE_EPSILON
    
    # 5. Scale by torso length
    lm_normalized = lm_centered / D_scale
    
    # Reshape back to (99,)
    return lm_normalized.reshape(-1)

def calculate_angles_from_landmarks(landmarks):
    """
    Calculate 12 joint angles from normalized landmarks
    Input: (99,) array -> Output: (12,) array
    """
    # Reshape to (33, 3)
    lm_reshaped = landmarks.reshape(NUM_LANDMARKS, 3)
    
    angles = np.zeros(ANGLES_DIM, dtype=np.float32)
    
    for chain_idx, (a, mid, c) in enumerate(JOINT_CHAINS):
        A = lm_reshaped[a]
        B = lm_reshaped[mid]
        C = lm_reshaped[c]
        
        # Vectors
        vecBA = A - B
        vecBC = C - B
        
        # Normalize vectors
        BA_norm = vecBA / (np.linalg.norm(vecBA) + 1e-6)
        BC_norm = vecBC / (np.linalg.norm(vecBC) + 1e-6)
        
        # Calculate angle
        dot = np.clip(np.dot(BA_norm, BC_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot)) / 180.0  # Normalize to [0, 1]
        
        angles[chain_idx] = angle
    
    return angles

def preprocess_frame(landmarks):
    """
    Full preprocessing pipeline for a single frame
    Input: Raw landmarks (99,) -> Output: Combined features (111,)
    """
    # 1. Relative normalization
    normalized_landmarks = relative_normalize_landmarks(landmarks)
    
    # 2. Calculate angles
    angles = calculate_angles_from_landmarks(normalized_landmarks)
    
    # 3. Combine: [12 angles | 99 normalized landmarks] = 111 features
    combined = np.concatenate([angles, normalized_landmarks])
    
    return combined

def preprocess_sequence(landmark_buffer, sequence_length):
    """
    Preprocess sequence of landmarks for model input
    Input: deque of raw landmarks -> Output: torch tensor (1, T, 111)
    """
    sequence = []
    
    for landmarks in landmark_buffer:
        if landmarks is not None:
            # Apply full preprocessing pipeline
            features = preprocess_frame(landmarks)
            sequence.append(features)
        else:
            # Padding for missing frames
            sequence.append(np.zeros(TOTAL_FEATURES, dtype=np.float32))
    
    sequence = np.array(sequence)
    
    # Pad if necessary
    if len(sequence) < sequence_length:
        padding = np.zeros((sequence_length - len(sequence), TOTAL_FEATURES))
        sequence = np.vstack([padding, sequence])
    
    return torch.from_numpy(sequence).unsqueeze(0).float()

def draw_landmarks(frame, results, mp_pose):
    """Draw pose landmarks on frame"""
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return frame

def draw_info_panel(frame, prediction_text, confidence, color):
    """Draw information panel on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Main prediction
    cv2.putText(frame, prediction_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Confidence bar
    bar_width = int((w - 40) * confidence)
    cv2.rectangle(frame, (20, 70), (20 + bar_width, 85), color, -1)
    cv2.rectangle(frame, (20, 70), (w - 20, 85), (255, 255, 255), 2)
    
    conf_text = f"{confidence:.1%}"
    cv2.putText(frame, conf_text, (w - 120, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# Main app
def main():
    st.title("🏋️ Live Exercise Classification")
    st.markdown("### Real-time pose-based exercise recognition with 111-feature preprocessing")
    st.markdown("---")
    
    # Load model
    model, label_map, checkpoint = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your model path in config.py")
        return
    
    # Initialize MediaPipe
    pose, mp_pose = init_mediapipe()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Display model info
    st.sidebar.subheader("📊 Model Performance")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Accuracy", f"{checkpoint.get('accuracy', 0):.1%}")
        st.metric("Precision", f"{checkpoint.get('precision', 0):.1%}")
    with col2:
        st.metric("Recall", f"{checkpoint.get('recall', 0):.1%}")
        st.metric("F1 Score", f"{checkpoint.get('f1', 0):.3f}")
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.subheader("🎛️ Inference Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence to show prediction"
    )
    
    sequence_length = st.sidebar.number_input(
        "Sequence Length (frames)",
        min_value=10,
        max_value=100,
        value=config.EXPECTED_SEQUENCE_LENGTH,
        step=5,
        help="Number of frames to analyze"
    )
    
    show_landmarks = st.sidebar.checkbox("Show Pose Landmarks", value=True)
    show_angles = st.sidebar.checkbox("Show Joint Angles", value=False)
    
    st.sidebar.markdown("---")
    
    # Exercise labels
    st.sidebar.subheader("🏷️ Exercise Classes")
    if isinstance(label_map, dict):
        class_names = list(label_map.keys())
    else:
        class_names = list(label_map)
    
    for i, exercise in enumerate(class_names):
        st.sidebar.text(f"  {i}: {exercise}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Features**: 12 angles + 99 relative landmarks = 111 total")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Live Predictions")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        if show_angles:
            st.subheader("📐 Joint Angles")
            angles_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        start_button = st.button("▶️ Start", use_container_width=True)
    with col_btn2:
        stop_button = st.button("⏹️ Stop", use_container_width=True)
    
    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    # Main loop
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("❌ Could not open camera. Please check your camera connection.")
            return
        
        landmark_buffer = deque(maxlen=sequence_length)
        frame_count = 0
        
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                results = pose.process(rgb_frame)
                
                # Extract raw landmarks (99 features)
                raw_landmarks = extract_landmarks(results)
                
                if raw_landmarks is not None:
                    landmark_buffer.append(raw_landmarks)
                    
                    # Make prediction when buffer is full
                    if len(landmark_buffer) == sequence_length:
                        with torch.no_grad():
                            # Preprocess sequence (applies normalization + angle calculation)
                            sequence = preprocess_sequence(landmark_buffer, sequence_length)
                            sequence = sequence.to(config.DEVICE)
                            
                            # Get prediction
                            logits = model(sequence)
                            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            predicted_class = np.argmax(probabilities)
                            confidence = probabilities[predicted_class]
                            
                            # Update predictions
                            with prediction_placeholder.container():
                                if confidence >= confidence_threshold:
                                    st.success(f"### **{class_names[predicted_class]}**")
                                else:
                                    st.warning("### **Uncertain - Keep Moving**")
                            
                            with confidence_placeholder.container():
                                st.metric("Confidence", f"{confidence:.1%}")
                            
                            # Show probability distribution
                            with chart_placeholder.container():
                                prob_dict = {class_names[i]: float(probabilities[i]) 
                                           for i in range(len(class_names))}
                                st.bar_chart(prob_dict)
                            
                            # Show angles if requested
                            if show_angles:
                                # Get the latest preprocessed frame
                                latest_features = preprocess_frame(raw_landmarks)
                                angles = latest_features[:12]  # First 12 values are angles
                                
                                angle_names = [
                                    "R Shoulder", "L Shoulder", "R Elbow", "L Elbow",
                                    "R Hip", "L Hip", "R Knee", "L Knee",
                                    "R Ankle", "L Ankle", "Torso", "Spine"
                                ]
                                
                                with angles_placeholder.container():
                                    angle_dict = {angle_names[i]: float(angles[i] * 180) 
                                                for i in range(len(angle_names))}
                                    st.bar_chart(angle_dict)
                            
                            # Prepare display text
                            if confidence >= confidence_threshold:
                                text = f"{class_names[predicted_class]}"
                                color = (0, 255, 0)  # Green
                            else:
                                text = "Uncertain"
                                color = (255, 165, 0)  # Orange
                            
                            # Draw info panel on frame
                            frame = draw_info_panel(frame, text, confidence, color)
                    else:
                        # Buffer filling
                        progress = len(landmark_buffer) / sequence_length
                        text = f"Buffering... {len(landmark_buffer)}/{sequence_length}"
                        cv2.putText(frame, text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    # No pose detected
                    cv2.putText(frame, "No pose detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw landmarks
                if show_landmarks:
                    frame = draw_landmarks(frame, results, mp_pose)
                
                # Draw frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Small delay to reduce CPU usage
                cv2.waitKey(1)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    else:
        st.info("👆 Click '▶️ Start' to begin live exercise classification")
        
        # Show example image
        st.markdown("### How it works:")
        st.markdown("""
        1. **Pose Detection**: MediaPipe extracts 33 body landmarks
        2. **Normalization**: Landmarks are centered to mid-hip and scaled by torso length
        3. **Angle Calculation**: 12 joint angles are computed from normalized landmarks
        4. **Feature Combination**: 12 angles + 99 normalized landmarks = 111 features
        5. **Classification**: TCN model analyzes the sequence to predict exercise type
        """)

if __name__ == "__main__":
    main()