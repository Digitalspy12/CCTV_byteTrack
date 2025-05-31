import streamlit as st
import numpy as np
import cv2
from src.pipeline import PersonReIDPipeline

# --- CONFIGURATION ---
# Hardcoded paths for demo; you can add file selectors later
CONFIG_PATH = "configs/config.yaml"
VIDEO_PATHS = [
    "data\input_videos\camerastock0.mp4",
    "data\input_videos\camerastock1.mp4"
]

# --- MAIN STREAMLIT APP ---
st.set_page_config(page_title="CCTV Detection & Tracking Stream", layout="wide")
st.title("CCTV Detection & Tracking Stream")

# Initialize pipeline (cache to avoid reloading on rerun)
@st.cache_resource
def get_pipeline(config_path):
    return PersonReIDPipeline(config_path)

pipeline = get_pipeline(CONFIG_PATH)

# Start streaming button
if st.button("Start Streaming"):
    frame_placeholders = {}
    for cam_id in range(len(VIDEO_PATHS)):
        frame_placeholders[cam_id] = st.empty()
    
    # Stream processed frames
    for vis_frames in pipeline.stream_processed_frames(VIDEO_PATHS):
        for cam_id, frame in vis_frames.items():
            # Convert BGR (OpenCV) to RGB for Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
            frame_placeholders[cam_id].image(rgb_frame, channels="RGB", caption=f"Camera {cam_id}")
        # Optional: add a small sleep to avoid UI freezing
        # time.sleep(0.01)
        # Streamlit auto-reruns, so no need for manual break 