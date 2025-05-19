# ClassifY: Image Classification Application

## Overview
ClassifY is a lightweight image classification application built using OpenCV, Streamlit, and TensorFlow. It leverages the MobileNetV2 pre-trained model to classify images into various categories. The application is designed to be user-friendly and efficient, making it suitable for mobile and edge devices.

## Features
- **Pre-trained Model**: Utilizes MobileNetV2, a lightweight convolutional neural network architecture.
- **Image Preprocessing**: Automatically preprocesses uploaded images to ensure compatibility with the model.
- **Top-3 Predictions**: Displays the top-3 predictions with their respective confidence scores.
- **Interactive UI**: Built with Streamlit for an intuitive and interactive user experience.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Classify
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run main.py
   ```
2. Open the application in your browser (usually at `http://localhost:8501`).
3. Upload an image in JPG, JPEG, or PNG format.
4. Click the "Classify" button to view the predictions.

## Code Explanation

### `load_model()`
Loads the MobileNetV2 pre-trained model with weights trained on the ImageNet dataset. This function ensures the model is ready for inference.

### `preprocess_image(image)`
Prepares the uploaded image for classification by:
- Converting it to a NumPy array.
- Handling grayscale and RGBA images by converting them to RGB format.
- Resizing the image to 224x224 pixels.
- Preprocessing the image using MobileNetV2's `preprocess_input` function.
- Adding a batch dimension to the image.

### `classify_image(model, image)`
Classifies the preprocessed image using the loaded model. It:
- Predicts the class probabilities for the image.
- Decodes the predictions to human-readable labels and confidence scores.
- Returns the top-3 predictions.

### `main()`
The main function that:
- Sets up the Streamlit page configuration.
- Displays the application title and description.
- Loads the cached MobileNetV2 model.
- Handles image uploads and displays the uploaded image.
- Classifies the image upon user interaction and displays the predictions.

## Dependencies
- **OpenCV**: For image processing.
- **NumPy**: For numerical operations.
- **Streamlit**: For building the web application.
- **TensorFlow**: For loading and using the MobileNetV2 model.
- **Pillow**: For handling image uploads.

## Notes
- MobileNetV2 is a lightweight model designed for mobile and edge devices, making it efficient and fast.
- The application uses Streamlit's caching mechanism to load the model only once, improving performance.

## Future Enhancements
- Add support for additional pre-trained models.
- Implement real-time image capture and classification.
- Enhance the UI with more customization options.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
