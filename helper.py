import os
import cv2
import numpy as np
import glob
from PIL import Image
from collections import Counter
from ultralytics import YOLO
import streamlit as st

def load_model(model_path):
    return YOLO(model_path)

def predict_image(model, image, confidence):
    try:
        image = image.convert("RGB")
        image_array = np.array(image)
        results = model.predict(image_array, conf=confidence)
        return results
    except Exception as e:
        st.error(f"Image prediction failed: {e}")
        return None

def plot_boxes(results):
    if not results:
        return None, {}
    res = results[0]
    plotted = res.plot()
    counts = Counter([res.names[int(cls)] for cls in res.boxes.cls]) if res.boxes else {}
    return plotted[:, :, ::-1], counts

def process_video(model, video_path, confidence, stop_key="run_video"):
    cap = cv2.VideoCapture(video_path)
    os.makedirs("output", exist_ok=True)

    for f in glob.glob("output/*.mp4"):
        os.remove(f)

    output_path = "output/output_1.mp4"
    suffix = 1
    while os.path.exists(output_path):
        suffix += 1
        output_path = f"output/output_{suffix}.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        if not st.session_state.get(stop_key, True):
            st.warning("Video detection stopped manually.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model.predict(frame, conf=confidence)
            plotted_frame = results[0].plot()
            out.write(plotted_frame)
        except Exception as e:
            st.error(f"Video frame processing error: {e}")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def process_webcam(model, conf):
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()

    while st.session_state.get("run_webcam", False):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model.predict(frame, conf=conf, verbose=False)
            plotted_frame = results[0].plot()
            st_frame.image(plotted_frame, channels="BGR", use_column_width=True)

            with st.expander("Live Detection Summary"):
                if results[0].boxes:
                    counts = Counter([results[0].names[int(cls)] for cls in results[0].boxes.cls])
                    for cls, count in counts.items():
                        st.write(f"**{cls}:** {count}")
                else:
                    st.write("No objects detected.")
        except Exception as e:
            st.error(f"Webcam error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
