import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

# Load the trained generator model
generator = tf.keras.models.load_model('generator.h5')

st.title('DCGAN Image Generation')

# Get the latent dimension from the generator model
latent_dim = generator.input_shape[1]

# Function to generate and display images
def generate_images(num_images=1):
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images * 127.5) + 127.5  # Denormalize the images

    for i in range(num_images):
        img = array_to_img(generated_images[i])
        st.image(img, caption=f'Generated Image {i+1}', use_column_width=True)

# Sidebar for user input
st.sidebar.title('Options')
num_images = st.sidebar.slider('Number of Images', 1, 10, 1, 1)

if st.sidebar.button('Generate Images'):
    generate_images(num_images)
