import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tempfile

mp_holistic = mp.solutions.holistic  # Holistic model

actions = ["halo","nama","saya"]
model = load_model('realtimeV12.h5')

threshold = 0.5

class VideoTransformer():
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.sentence = []

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    def transform(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        keypoints = self.extract_keypoints(results)

        try:
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-15:]
        except NameError:
            self.sequence = [keypoints]

        if len(self.sequence) == 15:
            try:
                res = model.predict(np.expand_dims(self.sequence, axis=0))[0]

                if res[np.argmax(res)] > threshold:
                    self.sentence.append(actions[np.argmax(res)])

                if len(self.sentence) > 1:
                    self.sentence = self.sentence[-1:]
            except Exception as e:
                print("No sentence detected:", str(e))

        return image_rgb

st.markdown(
    """
    <style>
    .bar {
        background-color: rgb(114, 134, 211);
        height: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="bar"></div>', unsafe_allow_html=True)

st.title("Sign Language Detection")

uploaded_file = st.file_uploader("Upload a video file (5 second max.)", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filepath = temp_file.name
    
    st.video(temp_filepath)

    video = cv2.VideoCapture(temp_filepath)

    video_transformer = VideoTransformer()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        transformed_frame = video_transformer.transform(frame)

    video.release()

    st.subheader("Detected Word:")

    st.write(' '.join(video_transformer.sentence))