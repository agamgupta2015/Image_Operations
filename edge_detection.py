from skimage import filters,feature
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def pil_to_numpy(image):
    return np.array(image)

def convert_to_grayscale(image):
    return image.convert('L')

def sobel_filter(image):
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image)
    sobel_img = filters.sobel(image_np)
    return sobel_img

def prewitt_filter(image):
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image)
    prewitt_img = filters.prewitt(image_np)
    return prewitt_img

def roberts_cross_operator(image): # [1,0] [0,-1]
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image)
    roberts_img = filters.roberts(image_np)
    return roberts_img

def canny_edge_detector(image, sigma=4, low_threshold=0.04, high_threshold=0.10):
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image)
    image_np = image_np.astype(np.float64)
    image_np /= 255.0
    canny_img = feature.canny(image_np, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    return Image.fromarray((canny_img * 255).astype(np.uint8))
    # return canny_img

def scharr_operator(image):  # [-3 0 3; -10 0 10; -3 0 3]
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image)
    scharr_img = filters.scharr(image_np)
    return scharr_img

def marr_hildreth_edge_detector(image, sigma=1.0, threshold=0.01):
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image)
    marr_hildreth_img = filters.gaussian(image_np, sigma=sigma)
    st.image(marr_hildreth_img, caption='Gaussian on Marr-Hildreth',  width=300)
    marr_hildreth_img = filters.laplace(marr_hildreth_img)
    
    marr_hildreth_img = np.where(np.abs(marr_hildreth_img) > threshold, 255, 0).astype(np.uint8)
    return marr_hildreth_img

def zero_crossing_edge_detector(image, threshold=0.1):
    image = convert_to_grayscale(image)
    image_np = pil_to_numpy(image) 
    zero_crossing_img = filters.sobel(image_np)
    zero_crossing_img = np.where(np.abs(zero_crossing_img) > threshold, 255, 0).astype(np.uint8)
    return zero_crossing_img


def edge_decection():
    st.markdown("<h1 style='text-align: center;'>EDGE DETECTION FILTERS </h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", 'tif'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        selected_function = st.selectbox('Select an Edge Detection Filter', ['Sobel Filter','Prewitt Filter','Robert Cross Operator','Canny Edge Detector','Scharr Operator',
                                                                    'Marr-Hildreth Edge Detector','Zero Crossing Edge Detector'])

        if selected_function == 'Canny Edge Detector':
            sigma = st.slider("Sigma", min_value=0.1, max_value=10.0, value=4.0)
            low_threshold = st.slider("Low Threshold", min_value=0.01, max_value=0.99, value=0.04)
            high_threshold = st.slider("High Threshold", min_value=0.01, max_value=0.99, value=0.10)

        elif selected_function == 'Marr-Hildreth Edge Detector':
            sigma = st.slider("Sigma", min_value=0.1, max_value=10.0, value=4.0)
            threshold = st.slider("Threshold", min_value=0.001, max_value=0.02, value=0.001)

        col1, col2 = st.columns(2)

        with col1:
            with st.expander('Original Image'):
                st.image(image, caption='Original Image', width=300)
                if st.button(f"Show Histogram"):
                        plt.hist(np.array(image).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

        with col2:
            if selected_function == 'Sobel Filter':
                with st.expander(selected_function):
                    sobel_img = sobel_filter(image)
                    st.image(sobel_img, caption='Sobel Filtered Image', width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(sobel_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

            elif selected_function == 'Prewitt Filter':
                with st.expander(selected_function):
                    prewitt_img = prewitt_filter(image)
                    st.image(prewitt_img, caption='Prewitt Filtered Image',  width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(prewitt_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

            elif selected_function == 'Robert Cross Operator':
                with st.expander(selected_function):
                    roberts_img = roberts_cross_operator(image)
                    st.image(roberts_img, caption='Roberts Cross Operator Filtered Image',  width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(roberts_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()
                
            elif selected_function == 'Canny Edge Detector':
                with st.expander(selected_function):
                    canny_img = canny_edge_detector(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
                    st.image(canny_img, caption='Canny Edge Detector Filtered Image',  width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(canny_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

            elif selected_function == 'Scharr Operator':
                with st.expander(selected_function):
                    scharr_img = scharr_operator(image)
                    st.image(scharr_img, caption='Scharr Operator Filtered Image',  width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(scharr_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

            elif selected_function == 'Marr-Hildreth Edge Detector':
                with st.expander(selected_function):
                    marr_hildreth_img = marr_hildreth_edge_detector(image,sigma=sigma,threshold=threshold)
                    st.image(marr_hildreth_img, caption='Marr-Hildreth Edge Detector Filtered Image',  width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(marr_hildreth_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

            elif selected_function == 'Zero Crossing Edge Detector':
                with st.expander(selected_function):
                    zero_crossing_img = zero_crossing_edge_detector(image)
                    st.image(zero_crossing_img, caption='Zero Crossing Edge Detector Filtered Image',  width=300)
                    if st.button(f"Show Histogram {selected_function}"):
                        plt.hist(np.array(zero_crossing_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()
