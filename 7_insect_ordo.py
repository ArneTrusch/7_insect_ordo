import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import requests
from io import BytesIO

 

st.title("7 insect ordo")
 
st.write("""
What order belongs this critter to? Input an image & find out!
""")

model = load_model("7_insect_ordo_model.h5")

image_source = st.selectbox("Please choose your image source:", ["Upload", "URL"])

if image_source == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
elif image_source == "URL":
    image_url = st.text_input("Enter image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_column_width=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not load image. Error: {e}")
        except IOError as e:
            st.error(f"Invalid image format. Error: {e}")



def predict(image):
    # Resize the image to the required size (300, 300)
    img = image.resize((300, 300))

    # Convert the PIL Image to a numpy array
    img = np.array(img, dtype=np.uint8)

    # Scaling the Image Array values between 0 and 1
    img = img / 255.0

    # Add a batch dimension
    img = np.expand_dims(img, axis=0)

    # Get the Predicted Label for the loaded Image
    p = model.predict(img)

    # Label array
    labels = {0: 'Bees', 1: 'Beetles', 2: 'Butterfly', 3: 'Cicada', 4: 'Dragonfly', 5: 'Grasshopper',
              6: 'Moth'}

    predicted_class = labels[np.argmax(p[0], axis=-1)]
    
    # Create a dictionary with class names and probabilities
    class_probabilities = {labels[i]: prob for i, prob in enumerate(p[0])}
    
    return predicted_class, class_probabilities

import matplotlib.pyplot as plt

def plot_probabilities(class_probabilities):
    fig, ax = plt.subplots()
    colors = ['gold', 'red', 'blue', 'black', 'brown', 'green', 'hotpink']
    index = np.arange(len(class_probabilities))
    plt.bar(index, class_probabilities.values(), color=colors)
    plt.xlabel('Labels', fontsize=8)
    plt.ylabel('Probability', fontsize=8)
    plt.xticks(index, class_probabilities.keys(), fontsize=8, rotation=20)
    plt.title('Probability for loaded image')
    return fig

if st.button("Classify"):
    if image is not None:
        predicted_class, class_probabilities = predict(image)
        st.success(f"Predicted class: {predicted_class}")
        st.pyplot(plot_probabilities(class_probabilities))
    else:
        st.error("Please upload an image or provide an image URL.")
