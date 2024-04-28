import streamlit as st
import cv2
import numpy as np
from PIL import Image

def dilation(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erosion(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def opening(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def hit_or_miss(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)

def skeletonization(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def top_hat(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def bottom_hat(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

def morphological_gradient(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def thinning(image):
    skel = np.zeros(image.shape, np.uint8)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_image = eroded.copy()

        if cv2.countNonZero(binary_image) == 0:
            done = True

    return skel

def thickening(image):
    thickened = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
    return thickened

def morphylogoy_operations():
    st.markdown("<h1 style='text-align: center;'>MORPHOLOGICAL OPERATIONS </h1>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 0)
        st.image(image, caption='Original Image', use_column_width=True)

        operation = st.selectbox("Select Morphological Operation", ["Dilation", "Erosion", "Opening", "Closing",
                                                                    "Hit-or-Miss Transform", "Skeletonization",
                                                                    "Top Hat", "Bottom Hat","Thinning","Thickening", "Morphological Gradient"])

        if operation == "Dilation":
            kernel_size = st.slider("Select kernel size for Dilation", 3, 15, 3)
            dilated_image = dilation(image, kernel_size=(kernel_size, kernel_size))
            st.image(dilated_image, caption="Dilated Image", use_column_width=True)

        elif operation == "Erosion":
            kernel_size = st.slider("Select kernel size for Erosion", 3, 15, 3)
            eroded_image = erosion(image, kernel_size=(kernel_size, kernel_size))
            st.image(eroded_image, caption="Eroded Image", use_column_width=True)

        elif operation == "Opening":
            kernel_size = st.slider("Select kernel size for Opening", 3, 15, 3)
            opened_image = opening(image, kernel_size=(kernel_size, kernel_size))
            st.image(opened_image, caption="Opened Image", use_column_width=True)

        elif operation == "Closing":
            kernel_size = st.slider("Select kernel size for Closing", 3, 15, 3)
            closed_image = closing(image, kernel_size=(kernel_size, kernel_size))
            st.image(closed_image, caption="Closed Image", use_column_width=True)

        elif operation == "Hit-or-Miss Transform":
            kernel = np.array([
                [ 0, 0,  0],
                [ 0, 1,  1],
                [ 0, -1, 0]
            ], dtype=np.int8)
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            hit_or_miss_image = hit_or_miss(binary_image, kernel)
            st.image(hit_or_miss_image, caption="Hit-or-Miss Transformed Image", use_column_width=True)

        elif operation == "Skeletonization":
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            skeletonized_image = skeletonization(binary_image)
            st.image(skeletonized_image, caption="Skeletonized Image", use_column_width=True)
        
        elif operation == "Top Hat":
            kernel_size = st.slider("Select kernel size for Top Hat", 3, 15, 3)
            top_hat_image = top_hat(image, kernel_size=(kernel_size, kernel_size))
            st.image(top_hat_image, caption="Top Hat Image", use_column_width=True)

        elif operation == "Bottom Hat":
            kernel_size = st.slider("Select kernel size for Bottom Hat", 3, 15, 3)
            bottom_hat_image = bottom_hat(image, kernel_size=(kernel_size, kernel_size))
            st.image(bottom_hat_image, caption="Bottom Hat Image", use_column_width=True)

        elif operation == "Morphological Gradient":
            kernel_size = st.slider("Select kernel size for Morphological Gradient", 3, 15, 3)
            gradient_image = morphological_gradient(image, kernel_size=(kernel_size, kernel_size))
            st.image(gradient_image, caption="Morphological Gradient Image", use_column_width=True)

        elif operation == "Thinning":
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            thinned_image = thinning(binary_image)
            st.image(thinned_image, caption="Thinned Image", use_column_width=True)

        elif operation == "Thickening":
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            thickened_image = thickening(binary_image)
            st.image(thickened_image, caption="Thickened Image", use_column_width=True)

