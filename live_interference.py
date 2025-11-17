import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time

# --- Paths & Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/classifier_best.pth")
NORM_STATS_PATH = Path("data/processed/normalization_stats.npz")
LABEL_MAP_PATH = Path("data/processed/label_map.npy")

# ================================
# Model definitions
# ================================
class PoseAutoencoder(nn.Module):
    def __init__(self, input_dim=99, seq_len=60, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_linear = nn.Linear(hidden_dim * 2, latent_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n_forward = h_n[-2]
        h_n_backward = h_n[-1]
        h_n = torch.cat([h_n_forward, h_n_backward], dim=1)
        return self.encoder_linear(h_n)

    def forward(self, x):
        return self.encode(x)


class ExerciseClassifier(nn.Module):
    def __init__(self, autoencoder, num_classes, hidden_dim=128, freeze_encoder=True):
        super().__init__()
        self.encoder = autoencoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        latent_dim = autoencoder.encoder_linear.out_features
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            z = self.encoder.encode(x)
        return self.classifier(z)

# ================================
# Load model, normalization stats, and labels
# ================================
def load_assets():
    stats = np.load(NORM_STATS_PATH, allow_pickle=True)
    mean = stats["mean"].astype(np.float32)
    std  = stats["std"].astype(np.float32)
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).tolist()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint.get("num_classes", len(label_map))

    autoencoder = PoseAutoencoder(
        input_dim=checkpoint.get("input_dim", 99),
        seq_len=checkpoint.get("seq_len", 60),
        latent_dim=checkpoint.get("latent_dim", 64)
    )

    model = ExerciseClassifier(autoencoder, num_classes=num_classes, freeze_encoder=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()

    print(f"✅ Model loaded ({num_classes} classes)")
    return model, mean, std, label_map

# ================================
# Angle calculation helper
# ================================
def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# ================================
# Main live inference + rep counting
# ================================
def main():
    model, mean, std, label_map = load_assets()
    SEQ_LEN = 60
    buffer = []

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("🚀 Starting live inference... Press 'q' to quit.")

    prediction_text = "..."
    confidence_text = ""
    
    # Rep tracking variables
    pushup_reps = 0
    pushup_dir = 0
    jj_reps = 0
    jj_dir = 0
    plank_timer = 0
    plank_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        lm_flat = [0.0]*99
        if results.pose_landmarks:
            lm_flat = []
            for lm in results.pose_landmarks.landmark:
                lm_flat.extend([lm.x, lm.y, lm.z])
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        buffer.append(lm_flat)
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

        # Predict exercise
        if len(buffer) == SEQ_LEN:
            sequence = np.array(buffer, dtype=np.float32)[np.newaxis, ...]
            sequence = (sequence - mean) / std
            with torch.no_grad():
                tensor = torch.from_numpy(sequence).float().to(DEVICE)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
            prediction_text = label_map[pred_idx]
            confidence_text = f"{conf:.2f}"
            
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Convert normalized coordinates to pixel positions
            left_sh = np.array([lm[11].x * frame.shape[1], lm[11].y * frame.shape[0]])
            right_sh = np.array([lm[12].x * frame.shape[1], lm[12].y * frame.shape[0]])
            nose     = np.array([lm[0].x  * frame.shape[1], lm[0].y  * frame.shape[0]])

            # Neck = midpoint between shoulders
            neck = ((left_sh + right_sh) / 2).astype(int)

            # --- Draw neckline ---
            # 1️⃣ Shoulder line
            cv2.line(frame, tuple(left_sh.astype(int)), tuple(right_sh.astype(int)), (0, 255, 0), 2)
            # 2️⃣ Neck line (neck → nose)
            cv2.line(frame, tuple(neck), tuple(nose.astype(int)), (0, 255, 255), 2)
            # 3️⃣ Diagonals: neck → left shoulder and neck → right shoulder
            cv2.line(frame, tuple(neck), tuple(left_sh.astype(int)), (255, 0, 0), 2)
            cv2.line(frame, tuple(neck), tuple(right_sh.astype(int)), (255, 0, 0), 2)

            # Optional: draw points for neck and shoulders
            cv2.circle(frame, tuple(neck), 5, (0, 0, 255), -1)      # neck in red
            cv2.circle(frame, tuple(left_sh.astype(int)), 5, (0, 255, 0), -1)   # left shoulder in green
            cv2.circle(frame, tuple(right_sh.astype(int)), 5, (0, 255, 0), -1)  # right shoulder in green


        # Repetition counting
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # --- Push-up ---
            if prediction_text == "push-up":
                left_elbow = angle([lm[11].x, lm[11].y, lm[11].z],
                                   [lm[13].x, lm[13].y, lm[13].z],
                                   [lm[15].x, lm[15].y, lm[15].z])
                right_elbow = angle([lm[12].x, lm[12].y, lm[12].z],
                                    [lm[14].x, lm[14].y, lm[14].z],
                                    [lm[16].x, lm[16].y, lm[16].z])
                avg_elbow = (left_elbow + right_elbow)/2

                if pushup_dir == 0 and avg_elbow < 90:
                    pushup_dir = 1
                    pushup_reps += 0.5
                elif pushup_dir == 1 and avg_elbow > 160:
                    pushup_dir = 0
                    pushup_reps += 0.5

            # --- Jumping Jack ---
            # --- Jumping Jack: Reps + Form Evaluation ---
        if prediction_text == "jumping jack":
            # Landmarks
            left_sh = [lm[11].x, lm[11].y]
            right_sh = [lm[12].x, lm[12].y]
            left_wr = [lm[15].x, lm[15].y]
            right_wr = [lm[16].x, lm[16].y]
            left_ankle = [lm[27].x, lm[27].y]
            right_ankle = [lm[28].x, lm[28].y]

            # Distances
            shoulder_dist = np.linalg.norm(np.array(left_sh) - np.array(right_sh))
            hand_dist = np.linalg.norm(np.array(left_wr) - np.array(right_wr))
            foot_dist = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

            # Dynamic thresholds
            open_threshold = 1.2 * shoulder_dist
            close_threshold = 0.5 * shoulder_dist
            min_hand_height = min(left_sh[1], right_sh[1]) - 0.1  # y decreases upward
            min_foot_dist = 1.2 * shoulder_dist

            # --- Rep counting ---
            if jj_dir == 0 and hand_dist > open_threshold:
                jj_dir = 1
                jj_reps += 0.5
            elif jj_dir == 1 and hand_dist < close_threshold:
                jj_dir = 0
                jj_reps += 0.5

            # --- Form evaluation ---
            wrong_form = False

            # Hands must be above shoulder
            if left_wr[1] > left_sh[1] or right_wr[1] > right_sh[1]:
                wrong_form = True
            # Feet must be wide enough
            if foot_dist < min_foot_dist:
                wrong_form = True

            # Only show warning if form is wrong
            if wrong_form:
                cv2.putText(frame, "⚠️ Fix your form!", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # --- Plank ---
            if prediction_text == "plank":
                if plank_start is None:
                    plank_start = time.time()
                plank_timer = time.time() - plank_start
            else:
                plank_start = None
                plank_timer = 0

        # Display info
        cv2.putText(frame, f"Exercise: {prediction_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Push-ups: {int(pushup_reps)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(frame, f"Jumping Jacks: {int(jj_reps)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(frame, f"Plank Time: {int(plank_timer)}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        cv2.imshow("Physio AI - Exercise + Reps", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
