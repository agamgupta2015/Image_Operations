import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def calculate_histogram(image):
    # Convert image to grayscale
    img_gray = image.convert('L')
    
    # Convert image to numpy array
    img_array = np.array(img_gray)
    
    # Calculate histogram
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    
    return hist, bins

def thresholding(image, threshold_value):
    # Convert image to grayscale
    img_gray = image.convert('L')
    
    # Convert image to numpy array
    img_array = np.array(img_gray)
    
    # Apply thresholding
    thresholded_img = np.where(img_array > threshold_value, 255, 0)
    
    return Image.fromarray(thresholded_img.astype(np.uint8))

def log_transformation(image, c):
    # Convert image to grayscale
    img_gray = image.convert('L')
    
    # Convert image to numpy array
    img_array = np.array(img_gray)
    
    # Apply log transformation
    log_transformed_img = c * np.log(1 + img_array)
    
    return Image.fromarray(log_transformed_img.astype(np.uint8))

def gamma_transformation(image, gamma):
    # Convert image to grayscale
    img_gray = image.convert('L')
    
    # Convert image to numpy array
    img_array = np.array(img_gray)
    
    # Apply gamma transformation
    gamma_transformed_img = np.power(img_array / 255, gamma) * 255
    
    return Image.fromarray(gamma_transformed_img.astype(np.uint8))

def histogram_equalization(image):
    # Convert image to grayscale
    img_gray = image.convert('L')
    
    # Convert image to numpy array
    img_array = np.array(img_gray)
    
    # Calculate histogram
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    
    # Perform histogram equalization
    equalized_img_array = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
    equalized_img_array = (equalized_img_array * 255).astype(np.uint8)
    equalized_img = Image.fromarray(equalized_img_array.reshape(img_array.shape))
    
    return equalized_img


def load_equalization():
    st.markdown("<h1 style='text-align: center;'>HISTOGRAM EQUALIZTION </h1>", unsafe_allow_html=True)    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1,col2 = st.columns(2)
        with col1:
            st.image(image, caption='Original Image', use_column_width=True)
        
        # Display original image and histogram
        with col2:
            orig_hist, orig_bins = calculate_histogram(image)
            plt.figure(figsize=(8,8))
            plt.title('Original Image Intensity Histogram')
            plt.xlabel('Pixel Values')
            plt.ylabel('Frequency')
            plt.bar(orig_bins[:-1], orig_hist, color='black')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        
        st.markdown('<br>',unsafe_allow_html=True)
        # Histogram Equalization
        st.subheader("Histogram Equalization")
        col1,col2 = st.columns(2)
        with col1:
            equalized_img = histogram_equalization(image)
            st.image(equalized_img, caption='Equalized Image', use_column_width=True)
        
        # Equalized Image Histogram
        with col2:
            equalized_hist, _ = calculate_histogram(equalized_img)
            plt.figure(figsize=(8, 8))
            plt.title('Equalized Image Intensity Histogram')
            plt.xlabel('Pixel Values')
            plt.ylabel('Frequency')
            plt.bar(orig_bins[:-1], equalized_hist, color='black')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()



def intensity_transformation():
    st.markdown("<h1 style='text-align: center;'>INTENSITY TRANSFORMATION </h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        st.image(image, caption='Original Image', width=300)

        # Radio buttons for different image processing functions
        selected_function = st.radio('Select an image processing function', ['Thresholding', 'Log Transformation', 'Gamma Transformation'])
        threshold_value = st.slider("Threshold Value", min_value=0, max_value=255, value=127)
        col1,col2 = st.columns(2)
        if selected_function == 'Thresholding':
            with col1:
                thresholded_img = thresholding(image, threshold_value)
                st.image(thresholded_img, caption='Thresholded Image', width=300)
            with col2:
            # Histogram for thresholded image
                thresholded_hist, thresholded_bins = calculate_histogram(thresholded_img)
                plt.figure(figsize=(8, 8))
                plt.title('Thresholded Image Intensity Histogram')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.bar(thresholded_bins[:-1], thresholded_hist, color='orange')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

        elif selected_function == 'Log Transformation':
            c_value = st.slider("C Value", min_value=1, max_value=100, value=1)
            with col1:
                log_transformed_img = log_transformation(image, c_value)
                st.image(log_transformed_img, caption='Log Transformed Image', width=300)
            
            # Histogram for log transformed image
            with col2:
                log_transformed_hist, log_transformed_bins = calculate_histogram(log_transformed_img)
                
                plt.figure(figsize=(8, 8))
                plt.title('Log Transformed Image Intensity Histogram')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.bar(log_transformed_bins[:-1], log_transformed_hist, color='green')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

        elif selected_function == 'Gamma Transformation':
            gamma_value = st.slider("Gamma Value", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            with col1:
                gamma_transformed_img = gamma_transformation(image, gamma_value)
                st.image(gamma_transformed_img, caption='Gamma Transformed Image', width=300)
            with col2:
            # Histogram for gamma transformed image
                gamma_transformed_hist, gamma_transformed_bins = calculate_histogram(gamma_transformed_img)
                plt.figure(figsize=(8, 8))
                plt.title('Gamma Transformed Image Intensity Histogram')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.bar(gamma_transformed_bins[:-1], gamma_transformed_hist, color='blue')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
