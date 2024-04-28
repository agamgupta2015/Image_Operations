import streamlit as st
import numpy as np
import cv2
from scipy import fftpack
from scipy.signal import butter, freqz, lfilter

def fourier_transform(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
    return magnitude_spectrum

def butterworth_filter(image, low_cutoff, high_cutoff, order):
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            mask[i, j] = 1 / (1 + ((distance * low_cutoff) * (distance * high_cutoff)) ** order)
    filtered_image = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(image) * mask)))
    return np.abs(filtered_image)

# Streamlit UI
st.title("Frequency Domain Filters")

# Upload image
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image is not None:
    image = cv2.imread(image, 0)

    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_column_width=True)

    # Fourier Transform
    st.subheader("Fourier Transform")
    ft_image = fourier_transform(image)
    st.image(ft_image, caption="Magnitude Spectrum", use_column_width=True)

    # Butterworth Filter
    st.subheader("Butterworth Filter")
    low_cutoff = st.slider("Low Cutoff", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    high_cutoff = st.slider("High Cutoff", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    order = st.slider("Order", min_value=1, max_value=10, value=2, step=1)
    filtered_image = butterworth_filter(image, low_cutoff, high_cutoff, order)
    st.image(filtered_image, caption="Filtered Image", use_column_width=True)
