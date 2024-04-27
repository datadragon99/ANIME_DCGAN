import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
from zipfile import ZipFile
from io import BytesIO

# Load the trained generator model
generator = tf.keras.models.load_model('gen.h5')

st.title('DCGAN Image Generation')

# Get the latent dimension from the generator model
latent_dim = generator.input_shape[1]

# Function to generate and display images
def generate_images(num_images=1, randomness_level=1.0):
    noise = tf.random.normal([num_images, latent_dim]) * randomness_level
    generated_images = generator(noise, training=False)
    generated_images = (generated_images * 127.5) + 127.5  # Denormalize the images
    return generated_images

# Function to download images as a ZIP file
def download_images(generated_images):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for i, img in enumerate(generated_images):
            img_data = array_to_img(img)
            file_name = f'image_{i+1}.png'
            img_buffer = BytesIO()
            img_data.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr(file_name, img_buffer.getvalue())

    zip_buffer.seek(0)
    st.download_button(
        label="Download Images",
        data=zip_buffer,
        file_name="generated_images.zip",
        mime="application/zip",
    )

# Sidebar for user input
st.sidebar.title('Options')
num_images = st.sidebar.slider('Number of Images', 1, 10, 1, 1)
randomness_level = st.sidebar.slider('Randomness Level', 0.0, 2.0, 1.0, 0.1)

if st.sidebar.button('Generate Images'):
    generated_images = generate_images(num_images, randomness_level)
    for i, img in enumerate(generated_images):
        img_data = array_to_img(img)
        st.image(img_data, caption=f'Generated Image {i+1}', use_column_width=True)

    download_images(generated_images)
