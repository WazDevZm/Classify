import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
# this is an mage clssifer built using opencv and streamlit
# Load the pre-trained model
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, # already trained model, light model
    # MobileNetV2 is a lightweight model, suitable for mobile and edge devices
    preprocess_input,
    decode_predictions,
)
#tensorflow is being used to load pre-trained model, we are not traing from scratch
from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model
# omobilente vs is a convolutional neural network architecture that is designed to be lightweight and efficient, making it suitable for mobile and edge devices. It is a variant of the original MobileNet architecture, which was introduced by Google in 2017. MobileNetV2 builds upon the success of its predecessor by introducing several improvements and optimizations.
def preprocess_image(image):
    # corecy frmat of the image to be passed to the model
    img = np.array(image) # cobnverts h imge into an arra of numbers
    img =cv2.resize(img, (224, 224)) # resize the image to 224x224 pixels
    img = preprocess_input(img)
    img - np.expand_dims(img, axis=0) # add a batch dimension, this is a bacth dimension
    
    