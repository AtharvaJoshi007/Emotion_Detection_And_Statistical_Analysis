# Emotion_Detection_And_Statistical_Analysis
# Custom Emotion Detection CNN Model



## Overview

This repository presents a custom Convolutional Neural Network (CNN) model for emotion detection. It encompasses both the training of the model and real-time emotion detection using a webcam or camera feed.

## Model Architecture

The emotion detection model boasts the following architecture:

```python
# Custom CNN Model Architecture
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
# Add more convolutional and fully connected layers as needed
# ...
model.add(Dense(output_class, activation='softmax'))

## Getting Started
Prerequisites
Python 3.x
Dependencies listed in requirements.txt
GPU (recommended) for faster training

##Installation
Install and/or import the required dependencies
os
numpy
cv2
pandas
matplotlib
tensorflow
warnings
tqdm


# Custom Emotion Detection CNN Model

![Emotion Detection](emotion_detection_image.png)

## Real-time Emotion Detection

In addition to training the custom CNN model for emotion detection, this repository also includes code for real-time emotion detection using a webcam or camera feed. The pre-trained model is loaded, and OpenCV is used to perform real-time face detection and emotion recognition.

## Usage

To run real-time emotion detection, follow these steps:

1. Ensure you have installed the required dependencies listed in `requirements.txt`.

2. Download the pre-trained model weights `AJ_EDM_Mk6.h5` generated using the model training code provided earlier.

3. Run the real-time emotion detection script:

```bash
python live_emotion_detection.py


Ensure you have installed the required dependencies and downloaded the pre-trained model weights (AJ_EDM_Mk6.h5).
Run the real-time emotion detection code

##Thresholds and Classification
The real-time emotion detection script calculates emotion percentages based on the detected frames. If the percentages for certain emotions exceed specific thresholds, the candidate is classified as "nervous"; otherwise, they are classified as "not nervous." You can adjust the thresholds and classification logic in the code to suit your specific needs.

##Emotion Distribution Analysis
The script also provides an analysis of the distribution of detected emotions and the classification results through interactive charts. These charts visualize the emotional states of candidates in real-time.

##Results and Analysis
Model Training
The model was trained for a total of 300 epochs, and the saved model weights can be found in AJ_EDM_Mk6.h5.

##Statistical Analysis with OpenCV
In the live_emotion_detection.py script, statistical analysis is performed using OpenCV to detect emotions in facial expressions.

##Model Performance
The model's performance was evaluated using various metrics. Please refer to the code for detailed performance analysis.

##Conclusion
This project demonstrates a custom emotion detection CNN model and its integration with OpenCV for real-time emotion detection in images and video streams. Future improvements and additional features may include real-time video emotion detection, a user-friendly GUI, and continuous model training for improved accuracy.

