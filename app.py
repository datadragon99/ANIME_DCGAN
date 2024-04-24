import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
import base64
from io import BytesIO

# Load the trained generator model
generator = tf.keras.models.load_model('generator.h5')

st.title('DCGAN Image Generation')

# Get the latent dimension from the generator model
latent_dim = generator.input_shape[1]

# Function to generate and display images
def generate_images(num_images=1, randomness_level=1.0):
    noise = tf.random.normal([num_images, latent_dim]) * randomness_level
    generated_images = generator(noise, training=False)
    generated_images = (generated_images * 127.5) + 127.5  # Denormalize the images
    return generated_images

# Function to download an image
def download_image(image, file_name):
    img = array_to_img(image)
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    base64_str = base64.b64encode(byte_im).decode('utf-8')
    href = f'<a href="data:image/png;base64,{base64_str}" download="{file_name}.png">Download {file_name}.png</a>'
    return href

# Sidebar for user input
st.sidebar.title('Options')
num_images = st.sidebar.slider('Number of Images', 1, 10, 5, 1)
randomness_level = st.sidebar.slider('Randomness Level', 0.0, 2.0, 1.0, 0.1)

if st.sidebar.button('Generate Images'):
    generated_images = generate_images(num_images, randomness_level)

    cols = st.columns(num_images)
    for i, col in enumerate(cols):
        col.image(generated_images[i], caption=f'Generated Image {i+1}', use_column_width=True)
        col.markdown(download_image(generated_images[i], f'image_{i+1}'), unsafe_allow_html=True)
