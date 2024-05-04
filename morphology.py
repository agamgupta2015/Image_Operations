import streamlit as st
import cv2
import numpy as np

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

def morphology_gradient(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


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
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(image, opening)

def bottom_hat(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.subtract(closing, image)

def morphylogoy_operations():
    st.title("Morphological Operations App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","tif"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 0)

        operation = st.selectbox("Select Morphological Operation", ["Dilation", "Erosion", "Opening", "Closing","Morphology Gradient"
                                                                    ,"Hit-or-Miss Transform", "Skeletonization", "Top Hat", "Bottom Hat"])

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

        elif operation == "Morphology Gradient":
            kernel_size = st.slider("Select kernel size for Morphology Gradient", 3, 15, 3)
            gradient_image = morphology_gradient(image, kernel_size=(kernel_size, kernel_size))
            st.image(gradient_image, caption="Morphology Gradient Image", use_column_width=True)

        elif operation == "Hit-or-Miss Transform":
            st.subheader("Hit-or-Miss Transform")
            kernel_size = st.slider("Select kernel size for Hit-or-Miss Transform", 3, 7, 3)
            st.write("Enter the kernel matrix (0 for don't care, 1 for foreground, -1 for background):")
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.int8)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    kernel[i, j] = st.number_input(f"Value at position ({i}, {j})", value=0, step=1)
            hit_or_miss_image = hit_or_miss(image, kernel)
            st.image(hit_or_miss_image, caption="Hit-or-Miss Transformed Image", use_column_width=True)

        elif operation == "Skeletonization":
            skeletonized_image = skeletonization(image)
            st.image(skeletonized_image, caption="Skeletonized Image", use_column_width=True)


        elif operation == "Top Hat":
            kernel_size = st.slider("Select kernel size for Top Hat", 3, 15, 3)
            top_hat_image = top_hat(image, kernel_size=(kernel_size, kernel_size))
            st.image(top_hat_image, caption="Top Hat Image", use_column_width=True)

        elif operation == "Bottom Hat":
            kernel_size = st.slider("Select kernel size for Bottom Hat", 3, 15, 3)
            bottom_hat_image = bottom_hat(image, kernel_size=(kernel_size, kernel_size))
            st.image(bottom_hat_image, caption="Bottom Hat Image", use_column_width=True)
