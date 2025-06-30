import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import sys

print(sys.version)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


class SquatCounter(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.down_frames = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Coordonnées jambe droite
            r_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h,
            ]
            r_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h,
            ]
            r_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h,
            ]

            # Coordonnées jambe gauche
            l_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h,
            ]
            l_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h,
            ]
            l_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h,
            ]

            # Calcul des angles pour chaque jambe
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
            avg_angle = (r_angle + l_angle) / 2

            # Vérification que le bassin est bien descendu
            avg_hip_y = (r_hip[1] + l_hip[1]) / 2
            avg_knee_y = (r_knee[1] + l_knee[1]) / 2
            is_low_enough = avg_hip_y > avg_knee_y

            # Logique squat robuste
            if avg_angle < 90 and is_low_enough:
                self.down_frames += 1
                if self.down_frames > 7:
                    self.stage = "down"
            elif avg_angle > 160:
                if self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                self.down_frames = 0

            # Affichage
            color = (0, 255, 0) if self.stage == "up" else (0, 0, 255)

            cv2.putText(
                image,
                f"Angle D: {int(r_angle)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"Angle G: {int(l_angle)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"Angle Moy: {int(avg_angle)}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f"Squats: {self.counter}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )

            mp_draw.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        return image


# Interface Streamlit
st.title("🏋️ Compteur de Squats - Détection par Posture Améliorée")
st.write(
    "Faites des squats devant votre webcam. Seuls les squats valides (profonds, complets, symétriques) sont comptés."
)

# Flux vidéo amélioré
webrtc_streamer(
    key="squat-counter",
    video_transformer_factory=SquatCounter,
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    },
    async_processing=True,
    video_html_attrs={
        "style": {"width": "100%", "height": "720px"},
        "autoPlay": True,
        "muted": True,
        "playsInline": True,
    },
)
