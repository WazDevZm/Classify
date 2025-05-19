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
# tensorflow is being used to load pre-trained model, we are not traing from scratch
from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet') 
    return model
# omobilente vs is a convolutional neural network architecture that is designed to be lightweight and efficient, making it suitable for mobile and edge devices. It is a variant of the original MobileNet architecture, which was introduced by Google in 2017. MobileNetV2 builds upon the success of its predecessor by introducing several improvements and optimizations.

def preprocess_image(image):
    # corecy frmat of the image to be passed to the model
    img = np.array(image) # cobnverts h imge into an arra of numbers

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (224, 224)) # resize the image to 224x224 pixels
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0) # add a batch dimension, this is a bacth dimension
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Classify", page_icon="🖼️", layout="centered")
    st.title("ClassifY🖼️")
    st.write("Upload an image to classify it🥳.")    

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True,
        )

        btn = st.button("Classify")
        if btn:
            with st.spinner("Classifying..."):
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions: # underscore in the loop is used to ignore the first value in the tuple
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()
