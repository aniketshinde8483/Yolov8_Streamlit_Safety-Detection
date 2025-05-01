# app.py
import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"

import streamlit as st
import settings
import helper
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Detection", page_icon="⛑️", layout="wide")
st.title("🦺👷‍♂️Universal Safety Detection👷‍♂️🦺 ")

# Sidebar
st.sidebar.header("Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
source_type = st.sidebar.radio("Select Source", settings.SOURCES_LIST)
detect_button = st.sidebar.button("Run Detection")

# Load model
model = helper.load_model(settings.MODEL_PATH)

# Image
if source_type == settings.SOURCES_LIST[0]:
    uploaded_images = st.sidebar.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images and detect_button:
        for i, image_file in enumerate(uploaded_images, start=1):
            st.subheader(f"Image {i}")
            col1, col2 = st.columns(2)
            image = Image.open(image_file)

            with col1:
                st.image(image, caption="Original", use_container_width=True)

            with col2:
                results = helper.predict_image(model, image, confidence)
                plotted, counts = helper.plot_boxes(results)
                st.image(plotted, caption="Detected", use_column_width=True)

                with st.expander("Results"):
                    if counts:
                        for cls, count in counts.items():
                            st.write(f"**{cls}:** {count}")
                    else:
                        st.write("No objects detected.")

# Video
if source_type == settings.SOURCES_LIST[1]:
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video and detect_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(uploaded_video.read())
            video_path = temp.name

        st.video(uploaded_video)
        st.write("Processing Video...")

        # Process the video using the helper function
        output_path = helper.process_video(model, video_path, confidence)
        if os.path.exists(output_path):
            st.success(f"Detection Completed! Video saved to: {output_path}")
            st.video(output_path)
        else:
            st.error("Error processing video.")

# Webcam
if source_type == settings.SOURCES_LIST[2]:
    if detect_button:
        st.info("Starting Webcam... Click Stop Webcam to end.")
        helper.process_webcam(model, confidence)
    
    elif source_type == settings.WEBCAM:
        if "run" not in st.session_state:
            st.session_state["run"] = False

        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")

        if start_button:
            st.session_state["run"] = True

        if stop_button:
            st.session_state["run"] = False

        if st.session_state["run"]:
            helper.process_webcam(model, confidence)

