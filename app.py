import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import threading
import time

# Model selection in the sidebar
st.sidebar.title("Model Selection")
model_options = {
    "YOLOv8 Nano": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
    "YOLOv8 XLarge": "yolov8x.pt",
    "YOLOv9 Nano": "yolov9n.pt",
    "YOLOv9 Small": "yolov9s.pt",
    "YOLOv9 Medium": "yolov9m.pt",
    "YOLOv9 Large": "yolov9l.pt",
    "YOLOv9 XLarge": "yolov9x.pt",
    "Custom Model": None  # Allow for custom model path
}
selected_model = st.sidebar.selectbox("Select a YOLOv8/v9 model", list(model_options.keys()))

if selected_model == "Custom Model":
    custom_model_path = st.sidebar.text_input("Enter custom model path")
    if custom_model_path:
        try:
            model = YOLO(custom_model_path)
        except Exception as e:
            st.sidebar.error(f"Error loading custom model: {e}")
            st.stop()  # Stop execution if custom model fails to load
    else:
        st.sidebar.warning("Please enter a custom model path.")
        st.stop()  # Stop execution until a path is provided
else:
    model_path = model_options[selected_model]
    model = YOLO(model_path)


# Input source selection
st.sidebar.title("Input Source")
input_source = st.sidebar.radio("Select Input Source", ("Image Upload", "Live Camera"))


def predict(chosen_model, img, classes=None, conf=0.5):
    results = chosen_model.predict(img, classes=classes, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=None, conf=0.5):
    img_copy = img.copy()
    results = predict(chosen_model, img_copy, classes, conf=conf)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            cls = int(box.cls)
            conf = float(box.conf[0])
            label = result.names[cls]

            x1, y1, x2, y2 = map(int, xyxy)

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (225, 0, 0), 2)
            cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    return img_copy, results


# Streamlit app
st.title("Object Detection with Ultralytics YOLOv8/v9")

if input_source == "Image Upload":
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        class_names = model.names
        selected_classes = st.multiselect("Select classes to detect (optional)", options=class_names)
        classes_indices = [class_names.index(c) for c in selected_classes] if selected_classes else None

        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

        result_img, results = predict_and_detect(model, orig_image, classes=classes_indices, conf=confidence_threshold)

        st.image(orig_image, caption="Original Image", use_container_width=True)
        st.image(result_img, caption="Detected Objects", use_container_width=True)

        if results and results[0].boxes:
            data = []
            for box in results[0].boxes:
                xyxy = box.xyxy[0]
                cls = int(box.cls)
                conf = float(box.conf[0])
                label = results[0].names[cls]
                x1, y1, x2, y2 = map(int, xyxy)
                data.append({"Class": label, "Confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            df = pd.DataFrame(data)
            st.dataframe(df)
        elif results and not results[0].boxes:
            st.write("No objects detected.")

elif input_source == "Live Camera":
    st.write("Live camera feed:")

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)  # 0 for default camera

    if not video_capture.isOpened():
        st.error("Could not open camera.")
        st.stop()

    # Create a placeholder for the image
    frame_placeholder = st.empty()

    # Thread to capture and process frames
    def process_frames():
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            result_img, results = predict_and_detect(model, frame)  # Process the frame
            frame_placeholder.image(result_img, use_container_width=True)  # Update the placeholder

            # Optional: Add a small delay for smoother display (adjust as needed)
            time.sleep(0.01)

    # Start the frame processing thread
    processing_thread = threading.Thread(target=process_frames, daemon=True) # Daemon thread allows app to exit
    processing_thread.start()

    # Keep the Streamlit app running (the thread will handle updates)
    while True:
        time.sleep(0.1)  # Keep the main thread alive (adjust as needed)
        if not processing_thread.is_alive(): # Check if processing thread is still running
            break

    video_capture.release() # Release the camera when done
    st.write("Camera feed stopped.")
