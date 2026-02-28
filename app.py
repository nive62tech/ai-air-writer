import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
import cv2
import numpy as np
import av

st.set_page_config(page_title="AI Air Writing", layout="centered")
st.title("✍️ AI Air Writing")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

if "canvas" not in st.session_state:
    st.session_state.canvas = None

if st.button("🧹 Clear Canvas"):
    st.session_state.canvas = None


def process_frame(image):
    if st.session_state.canvas is None:
        st.session_state.canvas = np.zeros_like(image)

    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            pt = hand_landmarks.landmark[8]
            h, w, _ = image.shape
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(st.session_state.canvas, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    return cv2.addWeighted(image, 0.7, st.session_state.canvas, 0.7, 0)


def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed = process_frame(img)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")


webrtc_streamer(
    key="air-writing",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)