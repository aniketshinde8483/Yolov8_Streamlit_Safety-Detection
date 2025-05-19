import os
import streamlit as st
import settings
import helper
from PIL import Image

os.environ["STREAMLIT_WATCH_FILE"] = "none"

st.set_page_config(page_title="PPE Detection", page_icon="⛑", layout="wide")
st.title("⛑🦺 PPE Detection with YOLOv8 🦺⛑")
st.markdown("Upload images, videos, or use webcam for PPE detection using YOLOv8.")

st.sidebar.header("Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
source_type = st.sidebar.radio("Select Source", settings.SOURCES_LIST)
detect_button = st.sidebar.button("Run Detection")

model = None
if detect_button:
    model = helper.load_model(settings.MODEL_PATH)

if source_type == settings.IMAGE:
    uploaded_images = st.sidebar.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_images and model:
        for i, image_file in enumerate(uploaded_images, start=1):
            st.subheader(f"Image {i}")
            col1, col2 = st.columns(2)
            image = Image.open(image_file)

            with col1:
                st.image(image, caption="Original", use_column_width=True)

            with col2:
                with st.spinner("Detecting objects..."):
                    results = helper.predict_image(model, image, confidence)
                    plotted, counts = helper.plot_boxes(results)
                    if plotted is not None:
                        st.image(plotted, caption="Detected", use_column_width=True)
                    else:
                        st.write("No detections.")

                with st.expander("Detection Results"):
                    if counts:
                        for cls, count in counts.items():
                            st.write(f"**{cls}:** {count}")
                    else:
                        st.write("No objects detected.")

elif source_type == settings.VIDEO:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        if st.sidebar.button("Start Video Detection"):
            st.session_state["run_video"] = True

        if st.sidebar.button("Stop Video Detection"):
            st.session_state["run_video"] = False

        if st.session_state.get("run_video", False) and model:
            st.info("Running detection on video...")
            with st.spinner("Processing..."):
                output_path = helper.process_video(model, "temp_video.mp4", confidence, stop_key="run_video")

            if output_path and os.path.exists(output_path):
                st.success("Video processed!")
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button("📥 Download Video", f, file_name=os.path.basename(output_path), mime="video/mp4")
            else:
                st.error("Failed to process video.")

elif source_type == settings.WEBCAM:
    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False

    if st.sidebar.button("Start Webcam"):
        st.session_state["run_webcam"] = True
    if st.sidebar.button("Stop Webcam"):
        st.session_state["run_webcam"] = False

    if st.session_state["run_webcam"] and model:
        st.info("Webcam running... Press 'Stop Webcam' to stop.")
        helper.process_webcam(model, confidence)
