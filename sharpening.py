from skimage import filters,feature
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import streamlit as st

def laplacian_filter(image):
    # Define the Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    # Apply the filter
    filtered_image = image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=3))
    return filtered_image

def unsharp_masking(image, sigma=1.0, strength=1.5):
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale if not already in that mode
    blurred = image.filter(ImageFilter.GaussianBlur(sigma))
    unsharp_image = ImageChops.difference(image, blurred)
    unsharp_image = ImageChops.add(image, unsharp_image, strength, 0)
    return unsharp_image

def high_boost_filter(image, sigma=1.0, boost_factor=2.0):
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale if not already in that mode
    blurred = image.filter(ImageFilter.GaussianBlur(sigma))
    sharpened = ImageChops.subtract(image, blurred)
    boosted = ImageChops.add(image, sharpened, boost_factor, 0)
    return boosted

def gradient_filter(image):
    # Define the kernel for gradient filter
    kernel = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
    # Apply the filter
    filtered_image = image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1))
    return filtered_image

def laplacian_of_gaussian(image, sigma=1.0):
    blurred = image.filter(ImageFilter.GaussianBlur(sigma))
    laplacian = laplacian_filter(blurred)
    return laplacian


def pil_to_numpy(image):
    return np.array(image)



def sharpening():
    st.markdown("<h1 style='text-align: center;'>SHARPENING FILTERS </h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        selected_function = st.selectbox('Select a sharpening filter', ['Laplacian Filter','Unsharp Masking','High Boost Filter','Gradient Filter','Laplacian Of Gaussian (LOG) filter'])
        st.image(image,caption='Original Image', use_column_width=True)
        if selected_function == 'Laplacian Filter':
            laplacian_img = laplacian_filter(image)
            st.image(laplacian_img, caption='Laplacian Filtered Image', use_column_width=True)

        elif selected_function == 'Unsharp Masking':
            unsharp_img = unsharp_masking(image)
            st.image(unsharp_img, caption='Unsharp Masking Filtered Image', use_column_width=True)

        elif selected_function == 'High Boost Filter':
            boosted_img = high_boost_filter(image)
            st.image(boosted_img, caption='High Boost Filtered Image', use_column_width=True)

        elif selected_function == 'Gradient Filter':
            gradient_img = gradient_filter(image)
            st.image(gradient_img, caption='Gradient Filtered Image', use_column_width=True)

        elif selected_function == 'Laplacian of Gaussian (LoG) Filter':
            log_img = laplacian_of_gaussian(image)
            st.image(log_img, caption='Laplacian of Gaussian (LoG) Filtered Image', use_column_width=True)

        