import streamlit as st
from PIL import Image
import numpy as np
import scipy.ndimage

# Define smoothing filter functions
def mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed_image = scipy.ndimage.convolve(image, kernel)
    return smoothed_image

def gaussian_filter(image, kernel_size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)), (kernel_size, kernel_size))
    kernel /= np.sum(kernel)
    smoothed_image = scipy.ndimage.convolve(image, kernel)
    return smoothed_image

def box_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed_image = scipy.ndimage.convolve(image, kernel)
    return smoothed_image

def median_filter(image, kernel_size):
    smoothed_image = scipy.ndimage.median_filter(image, size=kernel_size)
    return smoothed_image

# HTML header and CSS styles
html_header = """
    <style>
        h1, h2 {
            text-align: center;
        }
    </style>
"""

# Add HTML header and CSS
st.markdown(html_header, unsafe_allow_html=True)

def smoothing_filters():
    st.markdown("<h1 style='text-align: center;'>SMOOTHING FILTERS</h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('L') 
        selected_filter = st.selectbox('Select a smoothing filter', ['Mean Filter', 'Gaussian Filter', 'Box Filter', 'Median Filter'])
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
        sigma = st.slider("Sigma (Only for Gaussian)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        col1,col2 = st.columns(2)
        with col1: 
            st.image(image, caption='Original Image', width=300)
        with col2:
            if selected_filter == 'Mean Filter':
                smoothed_img = mean_filter(np.array(image), kernel_size)
                st.image(smoothed_img, caption='Mean Filtered Image', width=300)

            elif selected_filter == 'Gaussian Filter':
                # sigma = st.slider("Sigma", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
                smoothed_img = gaussian_filter(np.array(image), kernel_size, sigma)
                st.image(smoothed_img, caption='Gaussian Filtered Image', width=300)

            elif selected_filter == 'Box Filter':
                smoothed_img = box_filter(np.array(image), kernel_size)
                st.image(smoothed_img, caption='Box Filtered Image', width=300)

            elif selected_filter == 'Median Filter':
                smoothed_img = median_filter(np.array(image), kernel_size)
                st.image(smoothed_img, caption='Median Filtered Image', width=300)
    


