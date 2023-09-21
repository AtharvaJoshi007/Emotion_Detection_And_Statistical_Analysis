# Emotion_Detection_And_Statistical_Analysis
## Custom Emotion Detection CNN Model



## Overview

This repository presents a custom Convolutional Neural Network (CNN) model for emotion detection. It encompasses both the training of the model and real-time emotion detection using a webcam or camera feed.
## Necessary Imports

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
# matplotlib.inline
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img
```
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
model.add(Dense(output_class,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

#---------------TRAIN THE MODEL---------------#
history=model.fit(x=x_train,y=y_train,batch_size=512,epochs=300)
# save the model
model.save('AJ_EDM_Mk6.h5')
```

## Getting Started

Prerequisites
Python 3.x
Dependencies listed in requirements.txt
GPU (recommended) for faster training


## Real-time Emotion Detection

In addition to training the custom CNN model for emotion detection, this repository also includes code for real-time emotion detection using a webcam or camera feed. The pre-trained model is loaded, and OpenCV is used to perform real-time face detection and emotion recognition.

## Load the model 
``` python
# Load Saved Model from AJ_Model_EDM.py
model = tf.keras.models.load_model(r'AJ_EDM_Mk6.h5', compile=False)
model.compile(optimizer="adam", loss="categorical_cross_entropy", metrics='accuracy')
harr_cascade = cv2.CascadeClassifier("Harr_Face.xml")


# The following One-Hot-Encoder Codes are for the different emotions
preds = {"[1, 0, 0, 0, 0, 0, 0]": 'Angry',
         "[0, 1, 0, 0, 0, 0, 0]": 'DIsgust',
         "[0, 0, 1, 0, 0, 0, 0]": 'Fear',
         "[0, 0, 0, 1, 0, 0, 0]": 'Happy',
         "[0, 0, 0, 0, 1, 0, 0]": 'Neutral',
         "[0, 0, 0, 0, 0, 1, 0]": 'Sad',
         "[0, 0, 0, 0, 0, 0, 1]": 'Surprise',
         "[0, 0, 0, 0, 0, 0, 0]": 'NONE'}

emotion_counts = {'Angry': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
total_frames = 1
```

## Usage

To run real-time emotion detection, follow these steps:

1. Ensure you have installed the required dependencies listed in `requirements.txt`.

2. Download the pre-trained model weights `AJ_EDM_Mk6.h5` generated using the model training code provided earlier.

3. Run the real-time emotion detection script `python live_emotion_detection.py`

### Thresholds and Classification
The real-time emotion detection script calculates emotion percentages based on the detected frames. If the percentages for certain emotions exceed specific thresholds, the candidate is classified as "nervous"; otherwise, they are classified as "not nervous." You can adjust the thresholds and classification logic in the code to suit your specific needs.

### Emotion Distribution Analysis
The script also provides an analysis of the distribution of detected emotions and the classification results through interactive charts. These charts visualize the emotional states of candidates in real-time.

### Results and Analysis
Model Training
The model was trained for a total of 300 epochs, and the saved model weights can be found in AJ_EDM_Mk6.h5.

### Statistical Analysis with OpenCV
In the live_emotion_detection.py script, statistical analysis is performed using OpenCV to detect emotions in facial expressions.

### Model Performance
The model's performance was evaluated using various metrics. Please refer to the code for detailed performance analysis.

## Conclusion
This project demonstrates a custom emotion detection CNN model and its integration with OpenCV for real-time emotion detection in images and video streams. Future improvements and additional features may include real-time video emotion detection, a user-friendly GUI, and continuous model training for improved accuracy.

