# from skimage import filters,feature
# from PIL import Image, ImageFilter, ImageChops
# import numpy as np
# import streamlit as st

# def laplacian_filter(image):
#     # Define the Laplacian kernel
#     kernel = np.array([[0, 1, 0],
#                        [1, -4, 1],
#                        [0, 1, 0]])
#     # Apply the filter
#     filtered_image = image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=3))
#     return filtered_image

# def unsharp_masking(image, sigma=1.0, strength=0.5):
#     if image.mode != 'L':
#         image = image.convert('L')  # Convert to grayscale if not already in that mode
#     blurred = image.filter(ImageFilter.GaussianBlur(sigma))
#     unsharp_image = ImageChops.difference(image, blurred)
#     unsharp_image = ImageChops.add(image, unsharp_image, strength, 0)
#     return unsharp_image

# def high_boost_filter(image, sigma=1.0, boost_factor=2.0):
#     if image.mode != 'L':
#         image = image.convert('L')  # Convert to grayscale if not already in that mode
#     blurred = image.filter(ImageFilter.GaussianBlur(sigma))
#     sharpened = ImageChops.subtract(image, blurred)
#     boosted = ImageChops.add(image, sharpened, boost_factor, 0)
#     return boosted

# def gradient_filter(image):
#     # Define the kernel for gradient filter
#     kernel = np.array([[-1, 0, 1],
#                        [-1, 0, 1],
#                        [-1, 0, 1]])
#     # Apply the filter
#     filtered_image = image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1))
#     return filtered_image

# def pil_to_numpy(image):
#     return np.array(image)



# def sharpening():
#     st.markdown("<h1 style='text-align: center;'>SHARPENING FILTERS </h1>", unsafe_allow_html=True)
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])
    
#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         selected_function = st.selectbox('Select a sharpening filter', ['Laplacian Filter','Unsharp Masking','High Boost Filter','Gradient Filter'])
#         col1,col2 = st.columns(2)
#         with col1:
#             st.image(image,caption='Original Image', width=300)
#         with col2:
#             if selected_function == 'Laplacian Filter':
#                 laplacian_img = laplacian_filter(image)
#                 st.image(laplacian_img, caption='Laplacian Filtered Image', width=300)

#             elif selected_function == 'Unsharp Masking':
#                 unsharp_img = unsharp_masking(image)
#                 st.image(unsharp_img, caption='Unsharp Masking Filtered Image', width=300)

#             elif selected_function == 'High Boost Filter':
#                 boosted_img = high_boost_filter(image)
#                 st.image(boosted_img, caption='High Boost Filtered Image', width=300)

#             elif selected_function == 'Gradient Filter':
#                 gradient_img = gradient_filter(image)
#                 st.image(gradient_img, caption='Gradient Filtered Image', width=300)


# from skimage import filters, feature
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import streamlit as st
import re
import matplotlib.pyplot as plt

def laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    filtered_image = image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=3))
    return filtered_image

def unsharp_masking(image, sigma=1.0, strength=0.5):
    if image.mode != 'L':
        image = image.convert('L')
    blurred = image.filter(ImageFilter.GaussianBlur(sigma))
    unsharp_image = ImageChops.difference(image, blurred)
    unsharp_image = ImageChops.add(image, unsharp_image, strength, 0)
    return unsharp_image

def high_boost_filter(image, sigma=1.0, boost_factor=2.0):
    if image.mode != 'L':
        image = image.convert('L')
    blurred = image.filter(ImageFilter.GaussianBlur(sigma))
    sharpened = ImageChops.subtract(image, blurred)
    boosted = ImageChops.add(image, sharpened, boost_factor, 0)
    return boosted

def gradient_filter(image):
    kernel = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
    filtered_image = image.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1))
    return filtered_image

def pil_to_numpy(image):
    return np.array(image)

def apply_filter(image, filter_name):
    if filter_name == 'Laplacian Filter':
        return laplacian_filter(image)
    elif filter_name == 'Unsharp Masking':
        return unsharp_masking(image)
    elif filter_name == 'High Boost Filter':
        return high_boost_filter(image)
    elif filter_name == 'Gradient Filter':
        return gradient_filter(image)
    else:
        return image

# Extract filter names from user prompt
def extract_filter_names(prompt):
    filter_names = ['Laplacian Filter', 'Unsharp Masking', 'High Boost Filter', 'Gradient Filter']
    selected_filters = []
    for name in filter_names:
        if re.search(r'\b' + re.escape(name.lower()) + r'\b', prompt.lower()):
            selected_filters.append(name)
    return selected_filters

# Main 
def sharpening():
    st.title("Sharpening Filter Selection")

    # User can either select a filter from dropdown or enter a prompt
    option = st.radio("Choose an option:", ['Select from dropdown', 'Write a prompt'])

    if option == 'Select from dropdown':
        selected_filter = st.selectbox('Select a sharpening filter', ['Laplacian Filter', 'Unsharp Masking', 'High Boost Filter', 'Gradient Filter'])
        st.write("Selected filter:", selected_filter)
        user_prompt = selected_filter
    elif option == 'Write a prompt':
        user_prompt = st.text_input("Enter your prompt:")

    # Extract filter names from prompt
    selected_filters = extract_filter_names(user_prompt)
    if selected_filters:
        st.write("Selected filters:", selected_filters)

    if "help" in user_prompt.lower():
        st.info("Filters:\n" + "\n".join(['Laplacian Filter', 'Unsharp Masking', 'High Boost Filter', 'Gradient Filter']))

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
             with st.expander('Original Image'):
                st.image(image, caption='Original Image', width=300)
                if st.button(f"Show Histogram"):
                        plt.hist(np.array(image).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()
        with col2:
            count = 1
            for filter_name in selected_filters:
                with st.expander(filter_name):
                    filtered_img = apply_filter(image, filter_name)
                    st.image(filtered_img, caption=filter_name, width=300, use_column_width=True, output_format='JPEG')
                    if st.button(f"Show Histogram {filter_name}"):
                        plt.hist(np.array(filtered_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()
        # with col2:
        #     count = 1
        #     for filter_name in selected_filters:
        #             if count%2==0:
        #                 with col1:
        #                     filtered_img = apply_filter(image, filter_name)
        #                     st.image(filtered_img, caption=filter_name, width=300)
        #                     count = count+1
        #             else:
        #                 with col2:
        #                     filtered_img = apply_filter(image, filter_name)
        #                     st.image(filtered_img, caption=filter_name, width=300)
        #                     count = count+1
