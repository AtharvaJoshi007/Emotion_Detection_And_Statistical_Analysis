import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Load Saved Model from AJ_Model_EDM.py
model = tf.keras.models.load_model(r'AJ_EDM_Mk6.h5', compile=False)
model.compile(optimizer="adam", loss="categorical_cross_entropy", metrics='accuracy')

harr_cascade = cv2.CascadeClassifier("Harr_Face.xml")

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

anger_threshold = 30
neutral_threshold = 30
fear_threshold = 30
happy_threshold = 30
sad_threshold = 30
surprise_threshold = 30

nervous_count = 0
not_nervous_count = 0

color_dict = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 255),
    'Fear': (0, 255, 0),
    'Happy': (128, 128, 128),
    'Neutral': (255, 0, 0),
    'Sad': (255, 255, 0),
    'Surprise': (255, 255, 255),
    'NONE': (0, 0, 0)
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        resized_roi = cv2.resize(roi_gray, (48, 48))
        flattened_roi = np.reshape(resized_roi, (1, -1))
        flattened_roi = np.float32(flattened_roi)

        prediction = model.predict(flattened_roi.reshape(1, 48, 48, 1))
        pred_label = preds[str([int(i) for i in (prediction[0])])]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color=color_dict[pred_label])
        cv2.putText(frame, pred_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=(255, 255, 255))
        if(pred_label != 'NONE' and pred_label != 'Disgust'):
            total_frames += 1
            emotion_counts[pred_label] += 1
    emotion_percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counts.items()}

    if (
        emotion_percentages['Angry'] > anger_threshold
        or emotion_percentages['Neutral'] > neutral_threshold
        or emotion_percentages['Fear'] > fear_threshold
        or emotion_percentages['Sad'] > sad_threshold
        or emotion_percentages['Surprise'] > surprise_threshold
    ):
        candidate_classification = "nervous"
        nervous_count += 1
    else:
        candidate_classification = "not nervous"
        not_nervous_count += 1

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

colors_rgba = {emotion: (color[0] / 255, color[1] / 255, color[2] / 255) for emotion, color in color_dict.items()}

emotions = list(emotion_percentages.keys())
percentages = list(emotion_percentages.values())

plt.figure(figsize=(10, 5))
plt.bar(emotions, percentages, color=[colors_rgba[emotion] for emotion in emotions])
plt.xlabel('Emotions')
plt.ylabel('Percentage')
plt.title('Emotion Distribution')
plt.ylim([0, 100])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

total_candidates = nervous_count + not_nervous_count
nervous_percentage = (nervous_count / total_candidates) * 100
not_nervous_percentage = (not_nervous_count / total_candidates) * 100

plt.figure(figsize=(10, 5))
plt.bar(["Nervous", "Not Nervous"], [nervous_percentage, not_nervous_percentage], color=['red', 'green'])
plt.xlabel('Classification')
plt.ylabel('Percentage')
plt.title('Nervous vs. Not Nervous Classification')
plt.ylim([0, 100])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

cap.release()
cv2.destroyAllWindows()
