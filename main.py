# import streamlit as st
# from PIL import Image
# import cv2 
# import numpy as np
# import matplotlib.pyplot as plt
# from sharpening import sharpening
# from edge_detection import edge_decection
# from morph import morph
# from smoothing import smoothing_filters
# from intensity import intensity_transformation,load_equalization
# from realtime import realtime

# html_header = """
#     <div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">
#         <h1 style="color:#333;text-align:center;">Image Transformations</h1>
#     </div>
# """

# html_footer = """
#     <div style="position:fixed;left:0;bottom:0;width:100%;background-color:#f0f0f0;padding:10px;border-top:1px solid #ccc;text-align:center;">
#         <p style="color:#555;">Made with ❤️ by YourName</p>
#     </div>
# """

# css =''' 
#     <style>
#         .container {
#             padding: 20px;
#             margin-top: 20px;
#             border: 1px solid #ccc;
#             border-radius: 10px;
#             background-color: #f9f9f9;
#         }
#         .image-container {
#             display: flex;
#             justify-content: space-around;
#             margin-top: 20px;
#         }
#         .image-wrapper {
#             flex: 1;
#             padding: 10px;
#             border: 1px solid #ddd;
#             border-radius: 5px;
#             box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
#             background-color: #fff;
#         }
#         .image-wrapper img {
#             max-width: 100%;
#             height: auto;
#             border-radius: 5px;
#             border: 1px solid #ddd;
#         }
#         .slider-container {
#             margin-top: 20px;
#         }
#         .histogram-container {
#             display: flex;
#             justify-content: space-around;
#             margin-top: 20px;
#         }
#         .histogram-wrapper {
#             flex: 1;
#             padding: 10px;
#             border: 1px solid #ddd;
#             border-radius: 5px;
#             box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
#             background-color: #fff;
#         }
#         .histogram-wrapper img {
#             max-width: 100%;
#             height: auto;
#             border-radius: 5px;
#         }
#     </style>
#     '''

# def main():
#     selected_box = st.sidebar.selectbox(
#     'Choose one of the following',
#     ('Welcome','Intensity Transformation','Histogram Equalization','Smoothing Filters','Sharpening Filters','Edge Detection Filters','Morphological Operations','Real Time Edge-Detection',))

#     if selected_box=='Welcome':
#         welcome()
#     if selected_box=='Intensity Transformation':
#         intensity_transformation()
#     if selected_box=='Histogram Equalization':
#         load_equalization() 
#     if selected_box=='Sharpening Filters':
#         sharpening()
#     if selected_box=='Edge Detection Filters':
#         edge_decection()
#     if selected_box=='Morphological Operations':
#         morph()
#     if selected_box=='Smoothing Filters':
#         smoothing_filters()
#     if selected_box=='Real Time Edge-Detection':
#         realtime()
        



# def welcome():
#     st.header('Welcome to the World of Image Processing')
#     # st.image("your_logo.png", use_column_width=True)  # Add your logo or an image related to image processing
#     st.write("Welcome to our interactive web application for image processing. Here, you can upload images and apply various sophisticated techniques to manipulate and enhance them.")
    
#     # Add a brief description of what image processing is and its importance
    
#     st.write("Image processing involves techniques used to perform operations on an image to extract information or enhance its features. It finds applications in various fields such as medical imaging, surveillance, remote sensing, and more. By leveraging image processing techniques, we can extract valuable insights from images, make them visually appealing, or improve their quality.")
    
#     st.subheader("Explore Our Features")
#     st.write("Our web app offers a wide range of image processing techniques categorized into sections, each providing unique functionalities:")
    
#     # List the categories of image processing techniques with brief descriptions
#     st.markdown("- *Intensity Transformation*: Adjust the brightness and contrast of images using techniques like thresholding, log transformation, and gamma transformation.")
#     st.markdown("- *Histogram Equalization*: Enhance the contrast of images by redistributing intensity values using histogram equalization techniques.")
#     st.markdown("- *Smoothing Filters*: Reduce noise and blur images using various smoothing filters such as Gaussian blur and median blur.")
#     st.markdown("- *Sharpening Filters*: Enhance the sharpness of images by applying sharpening filters like Laplacian and high-pass filters.")
#     st.markdown("- *Edge Detection Filters*: Detect edges and boundaries in images using algorithms like Sobel, Prewitt, and Canny edge detection.")
#     st.markdown("- *Frequency Domain Filters*: Perform operations in the frequency domain to filter images, remove noise, and enhance features.")
#     st.markdown("- *Morphological Filters*: Analyze and manipulate the geometric structures of images using morphological operations like dilation, erosion, opening, and closing.")
    
#     # Add more descriptive content or examples for each category if necessary
    
#     st.subheader("Get Started")
#     st.write("Ready to explore the fascinating world of image processing? Upload your images and start experimenting with our powerful tools!")
    
#     st.markdown(
#         f'<div style="position:fixed;left:0;bottom:0;width:100%;background-color:#f0f0f0;padding:10px;border-top:1px solid #ccc;text-align:center;">'
#         f'<p style="color:#555;">Made with ❤</p>'
#         f'</div>',
#         unsafe_allow_html=True
#     )
    
    
# if __name__ == "__main__":
#     main()


import streamlit as st
from PIL import Image
# import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sharpening import sharpening
from edge_detection import edge_decection
from morph import morph
from smoothing import smoothing_filters
from intensity import intensity_transformation, load_equalization
# from realtime import realtime

html_header = """
    <div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">
        <h1 style="color:#333;text-align:center;">Image Transformations</h1>
    </div>
"""

html_footer = """
    <div style="position:fixed;left:0;bottom:0;width:100%;background-color:#f0f0f0;padding:10px;border-top:1px solid #ccc;text-align:center;">
        <p style="color:#555;">Made by TriFusion</p>
    </div>
"""

css =''' 
    <style>
        .container {
            padding: 20px;
            margin-top: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .image-container {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .image-wrapper {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
            background-color: #fff;
        }
        .image-wrapper img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .slider-container {
            margin-top: 20px;
        }
        .histogram-container {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .histogram-wrapper {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
            background-color: #fff.
        }
        .histogram-wrapper img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
    </style>
    '''
selected_box = st.sidebar.selectbox(
        'Choose one of the following',
        ('Welcome', 'Intensity Transformation', 'Histogram Equalization', 'Smoothing Filters', 'Sharpening Filters', 'Edge Detection Filters', 'Morphological Operations')
)

def main():
    # st.markdown(html_header, unsafe_allow_html=True)
    # st.sidebar.title("For Help")
    if selected_box == 'Welcome':
        welcome()
    elif selected_box == 'Intensity Transformation':
        intensity_transformation()
    elif selected_box == 'Histogram Equalization':
        load_equalization() 
    elif selected_box == 'Sharpening Filters':
        sharpening()
    elif selected_box == 'Edge Detection Filters':
        edge_decection()
    elif selected_box == 'Morphological Operations':
        morph()
    elif selected_box == 'Smoothing Filters':
        smoothing_filters()
    # elif selected_box == 'Real Time Edge-Detection':
    #     realtime()

    st.markdown(html_footer, unsafe_allow_html=True)

    chatbot()

def welcome():
    st.header('Welcome to the World of Image Processing')
    st.write("Welcome to our interactive web application for image processing. Here, you can upload images and apply various sophisticated techniques to manipulate and enhance them.")
    st.write("Image processing involves techniques used to perform operations on an image to extract information or enhance its features. It finds applications in various fields such as medical imaging, surveillance, remote sensing, and more. By leveraging image processing techniques, we can extract valuable insights from images, make them visually appealing, or improve their quality.")
    st.subheader("Explore Our Features")
    st.write("Our web app offers a wide range of image processing techniques categorized into sections, each providing unique functionalities:")
    st.markdown("- *Intensity Transformation*: Adjust the brightness and contrast of images using techniques like thresholding, log transformation, and gamma transformation.")
    st.markdown("- *Histogram Equalization*: Enhance the contrast of images by redistributing intensity values using histogram equalization techniques.")
    st.markdown("- *Smoothing Filters*: Reduce noise and blur images using various smoothing filters such as Gaussian blur and median blur.")
    st.markdown("- *Sharpening Filters*: Enhance the sharpness of images by applying sharpening filters like Laplacian and high-pass filters.")
    st.markdown("- *Edge Detection Filters*: Detect edges and boundaries in images using algorithms like Sobel, Prewitt, and Canny edge detection.")
    st.markdown("- *Frequency Domain Filters*: Perform operations in the frequency domain to filter images, remove noise, and enhance features.")
    st.markdown("- *Morphological Filters*: Analyze and manipulate the geometric structures of images using morphological operations like dilation, erosion, opening, and closing.")
    st.subheader("Get Started")
    st.write("Ready to explore the fascinating world of image processing? Upload your images and start experimenting with our powerful tools!")

def chatbot():
    st.sidebar.subheader("Chatbot")
    user_input = st.sidebar.text_input("Ask me anything about image processing:")
    if user_input:
        response = handle_user_input(user_input)
        st.sidebar.write(response)

def handle_user_input(user_input):
    # Simple keyword matching for navigation
    if "intensity" in user_input.lower():
        return "- Thresholding\n- Log Transformation\n- Gamma Transformation"
    elif "histogram" in user_input.lower():
        return "- Histogram Equalization"
    elif "smoothing" in user_input.lower():
        return "- Mean filter\n- Median Filter\n- Gaussian Filter"
    elif "sharpening" in user_input.lower():
        return "- Laplacian Filter\n- Unsharp Masking\n- High Boost Filter"
    elif "edge" in user_input.lower():
        return "- Sobel Filter\n- Robert filter\n- Canny Edge Detector\n - Many more........"
    elif "morphological" in user_input.lower():
        return "- Opening\n- Closing\n- Dilation\n- Top-Hat\n- Skeleton"
    elif "help" in user_input.lower():
        return "You may perform different types of Image-Operations" 
    else:
        return "Work under process"

if __name__ == "__main__":
    main()


    
    