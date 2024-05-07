import streamlit as st
from PIL import Image
import numpy as np
import scipy.ndimage
import re
import matplotlib.pyplot as plt

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

def median_filter(image, kernel_size):
    smoothed_image = scipy.ndimage.median_filter(image, size=kernel_size)
    return smoothed_image


def apply_filter(image, filter_name,kernel,sigma):
    if filter_name == 'Mean Filter':
        return mean_filter(image,kernel)
    elif filter_name == 'Gaussian Filter':
        return gaussian_filter(image,kernel,sigma)
    elif filter_name == 'Median Filter':
        return median_filter(image,kernel)
    else:
        return image

# Extract filter names from user prompt
def extract_filter_names(prompt):
    filter_names = ['Mean Filter', 'Gaussian Filter','Median Filter']
    selected_filters = []
    for name in filter_names:
        if re.search(r'\b' + re.escape(name.lower()) + r'\b', prompt.lower()):
            selected_filters.append(name)
    return selected_filters

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

    option = st.radio("Choose an option:", ['Select from dropdown', 'Write a prompt'])

    if option == 'Select from dropdown':
        selected_filter = st.selectbox('Select a sharpening filter', ['Mean Filter', 'Gaussian Filter','Median Filter'])
        st.write("Selected filter:", selected_filter)
        user_prompt = selected_filter
    elif option == 'Write a prompt':
        user_prompt = st.text_input("Enter your prompt:")

    selected_filters = extract_filter_names(user_prompt)
    if selected_filters:
        st.write("Selected filters:", selected_filters)

    if "help" in user_prompt.lower():
        st.info("Filters:\n" + "\n".join(['Mean Filter', 'Gaussian Filter','Median Filter'])) 
       
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])

    kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
    sigma = st.slider("Sigma (Only for Gaussian)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('L') 
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Original Image', width=300)
        
        with col2:
            count = 1
            for filter_name in selected_filters:
                with st.expander(filter_name):
                    filtered_img = apply_filter(image, filter_name,kernel=kernel_size,sigma=sigma)
                    st.image(filtered_img, caption=filter_name, width=300, use_column_width=True, output_format='JPEG')
                    if st.button(f"Show Histogram {filter_name}"):
                        plt.hist(np.array(filtered_img).flatten(), bins=256, color='blue', alpha=0.7)
                        st.pyplot()

# import streamlit as st
# from PIL import Image
# import numpy as np
# import scipy.ndimage
# import re
# import matplotlib.pyplot as plt

# # Define smoothing filter functions
# def mean_filter(image, kernel_size):
#     kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
#     smoothed_image = scipy.ndimage.convolve(image, kernel)
#     return smoothed_image

# def gaussian_filter(image, kernel_size, sigma):
#     kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)), (kernel_size, kernel_size))
#     kernel /= np.sum(kernel)
#     smoothed_image = scipy.ndimage.convolve(image, kernel)
#     return smoothed_image

# def median_filter(image, kernel_size):
#     smoothed_image = scipy.ndimage.median_filter(image, size=kernel_size)
#     return smoothed_image


# def apply_filter(image, filter_name,kernel,sigma):
#     if filter_name == 'Mean Filter':
#         return mean_filter(image,kernel)
#     elif filter_name == 'Gaussian Filter':
#         return gaussian_filter(image,kernel,sigma)
#     elif filter_name == 'Median Filter':
#         return median_filter(image,kernel)
#     else:
#         return image

# # Extract filter names from user prompt
# def extract_filter_names(prompt):
#     filter_names = ['Mean Filter', 'Gaussian Filter','Median Filter']
#     selected_filters = []
#     for name in filter_names:
#         if re.search(r'\b' + re.escape(name.lower()) + r'\b', prompt.lower()):
#             selected_filters.append(name)
#     return selected_filters

# # HTML header and CSS styles
# html_header = """
#     <style>
#         h1, h2 {
#             text-align: center;
#         }
#     </style>
# """

# # Add HTML header and CSS
# st.markdown(html_header, unsafe_allow_html=True)

# def smoothing_filters():
#     st.markdown("<h1 style='text-align: center;'>SMOOTHING FILTERS</h1>", unsafe_allow_html=True)

#     option = st.radio("Choose an option:", ['Select from dropdown', 'Write a prompt'])

#     if option == 'Select from dropdown':
#         selected_filter = st.selectbox('Select a sharpening filter', ['Mean Filter', 'Gaussian Filter','Median Filter'])
#         st.write("Selected filter:", selected_filter)
#         user_prompt = selected_filter
#     elif option == 'Write a prompt':
#         user_prompt = st.text_input("Enter your prompt:")

#     selected_filters = extract_filter_names(user_prompt)
#     if selected_filters:
#         st.write("Selected filters:", selected_filters)

#     if "help" in user_prompt.lower():
#         st.info("Filters:\n" + "\n".join(['Mean Filter', 'Gaussian Filter','Median Filter'])) 
       
#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",'tif'])

#     kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
#     sigma = st.slider("Sigma (Only for Gaussian)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image).convert('L') 
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption='Original Image', width=300)
        
#         with col2:
#             count = 1
#             for filter_name in selected_filters:
#                     if count%2==0:
#                         with col1:
#                             filtered_img = apply_filter(image, filter_name,kernel=kernel_size,sigma=sigma)
#                             st.image(filtered_img, caption=filter_name, width=300)
#                             # if st.button("Show Histogram"):
#                             #     plt.hist(np.array(filtered_img).flatten(), bins=256, color='blue', alpha=0.7)
#                             #     st.pyplot()
#                             count = count+1
#                     else:
#                         with col2:
#                             filtered_img = apply_filter(image, filter_name,kernel=kernel_size,sigma=sigma)
#                             st.image(filtered_img, caption=filter_name, width=300)
#                             count = count+1
    # if uploaded_image is not None:
    #     image = Image.open(uploaded_image).convert('L') 
    #     selected_filter = st.selectbox('Select a smoothing filter', ['Mean Filter', 'Gaussian Filter','Median Filter'])
    #     kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
    #     sigma = st.slider("Sigma (Only for Gaussian)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
    #     col1,col2 = st.columns(2)
    #     with col1: 
    #         st.image(image, caption='Original Image', width=300)
    #     with col2:
    #         if selected_filter == 'Mean Filter':
    #             smoothed_img = mean_filter(np.array(image), kernel_size)
    #             st.image(smoothed_img, caption='Mean Filtered Image', width=300)

    #         elif selected_filter == 'Gaussian Filter':
    #             smoothed_img = gaussian_filter(np.array(image), kernel_size, sigma)
    #             st.image(smoothed_img, caption='Gaussian Filtered Image', width=300)

    #         elif selected_filter == 'Median Filter':
    #             smoothed_img = median_filter(np.array(image), kernel_size)
    #             st.image(smoothed_img, caption='Median Filtered Image', width=300)
    


