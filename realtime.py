import streamlit as st
import cv2
import numpy as np
from threading import Thread, Lock
import queue

class Filters:
    def __init__(self):
        pass

    def canny_edge_detection(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def capture_frames(cap, frame_queue, lock):
    while True:
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                with lock:
                    frame_queue.put(frame)

def realtime():
    st.markdown("<h1 style='text-align: center;'>REAL TIME DETECTION </h1>", unsafe_allow_html=True)

    # Menu with filter options
    filter_options = ["Original", "Canny Edge Detection"]
    selected_filter = st.sidebar.radio("Select Filter", filter_options)

    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    
    # Initialize Filters object
    filter = Filters()

    # Placeholder for the displayed image
    displayed_image = st.empty()

    # Queue for frames from the camera
    frame_queue = queue.Queue(maxsize=1)

    # Lock for synchronizing access to the frame queue
    lock = Lock()

    # Start the thread for capturing frames
    capture_thread = Thread(target=capture_frames, args=(cap, frame_queue, lock))
    capture_thread.start()

    # Main loop for real-time image processing
    while True:
        # Get frame from the queue
        try:
            with lock:
                frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        # Apply selected filter
        if selected_filter == "Canny Edge Detection":
            filtered_frame = filter.canny_edge_detection(frame)
        else:
            filtered_frame = frame

        # Display the filtered frame
        displayed_image.image(filtered_frame, channels="BGR")

    # Release the camera when the app is closed
    cap.release()

