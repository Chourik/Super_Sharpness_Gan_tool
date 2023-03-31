###Imports
import streamlit as st
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize

from sharpness_model import *

### Paths 
PATH_DARK_IMAGE = r"noir.jpg"
generator_path = r"gen_model_2_1.h5"

model = load_model(generator_path, custom_objects={'loss_function': vgg_loss_wrapper(image_shape)})
model.trainable = False

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file,'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; 
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def process_image(image):
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_norm = (image_array - 127.5) / 127.5
    # Resize the image
    shape_test = (96, 96, 3)  # Replace with the shape you used for training
    image_processed = resize(image_array, shape_test, mode='reflect', anti_aliasing=True)
    image_processed_norm = resize(image_norm, shape_test, mode='reflect', anti_aliasing=True)
    return image_processed_norm, image_processed


def display_result(image_processed, generated_image):
    # Display original image
    st.subheader("Original Image")
    st.image(image_processed, use_column_width=True,clamp = True)

    # Display generated image
    st.subheader("Generated Image")
    st.image(generated_image, use_column_width=True,clamp = True)

def denormalize(output):
    return (output * 127.5) + 127.5


st.markdown('<h1 style="color:white;">Super Sharpening AI-TOOL</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:white;">Created with Generative Adversarial Networks</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:white;">Drop your image to remove bluriness or increase quality </h3>', unsafe_allow_html=True)

#background image
#set_png_as_page_bg(PATH_DARK_IMAGE)

#image upload & processing
uploaded_file = st.file_uploader('Insert image for classification', type = ['png','jpg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Process the uploaded image
    image_processed_norm, image_processed = process_image(image)

    st.image(image_processed,clamp = True)
    # Run the processed image through the neural network

    gen_img = model.predict(np.expand_dims(image_processed_norm, axis=0))

    st.markdown('algo a fini de touner')
    # Denormalize the generated image
    # Denormalize the generated image
    generated_image = np.array(gen_img[0])
    generated_image_denormalized = denormalize(generated_image)

    # Display the denormalized image before clipping
    #st.image(generated_image_denormalized, caption="Denormalized image before clipping")

    # Clip values to the 0-255 range and convert to the appropriate data type
    denormalized_output = np.clip(generated_image_denormalized, 0, 255).astype(np.uint8)
    st.markdown(denormalized_output)
    # Display the results
    display_result(image_processed, denormalized_output)



#if __name__ == __main__():


# def process_image(image):
#     # Convert to numpy array and normalize
#     image_array = np.array(image)
#     image_norm = (image_array - 127.5) / 127.5
#     # Resize the image
#     shape_test = (96, 96, 3)  # Replace with the shape you used for training
#     image_processed = resize(image_norm, shape_test, mode='reflect', anti_aliasing=True)
#     #image_batch_lr = np.expand_dims(image_processed, axis=0)
#     return image_processed

    # generated_image = np.array(gen_img[0])
    # generated_image_denormalized = denormalize(generated_image)
    # # Clip values to the 0-255 range and convert to the appropriate data type
    # denormalized_output = np.clip(generated_image_denormalized, 0, 255).astype(np.uint8)