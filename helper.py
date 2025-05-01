# helper.py

import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import streamlit as st

import urllib.request
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)


def load_model(model_path):
    return YOLO(model_path)

def predict_image(model, image, confidence):
    image = image.convert("RGB")  # Ensure image has 3 channels
    image_array = np.array(image)
    results = model.predict(image_array, conf=confidence)
    return results

def plot_boxes(results):
    res = results[0]
    plotted = res.plot()
    counts = Counter([res.names[int(cls)] for cls in res.boxes.cls]) if res.boxes else {}
    return plotted[:, :, ::-1], counts  # RGB output

# helper.py

import os

def process_video(model, video_path, confidence):
    cap = cv2.VideoCapture(video_path)

    # Ensure the output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename
    base_name = "output"
    suffix = 1
    while os.path.exists(os.path.join(output_dir, f"{base_name}_{suffix}.mp4")):
        suffix += 1
    output_path = os.path.join(output_dir, f"{base_name}_{suffix}.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict on the frame
        results = model.predict(frame, conf=confidence)
        res = results[0]

        # Plot the results and ensure the frame is in BGR format
        plotted_frame = res.plot()
        out.write(plotted_frame)

    cap.release()
    out.release()
    return output_path

def process_webcam(model, conf):
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()

    while st.session_state["run"]:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        plotted_frame = results[0].plot()
        st_frame.image(plotted_frame, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()
