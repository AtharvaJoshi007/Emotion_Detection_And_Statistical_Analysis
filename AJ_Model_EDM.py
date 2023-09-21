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
def rescaleframe(frame,scale=1):
    width=int(frame.shape[1]* scale) #shape[1] refers to width
    height =int(frame.shape[0] *scale) #shape[0] refers to height
    dimensions=(width,height) # a tuple containing width and height
    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)
# Load Dataset
Train_Dir =r'/content/train'
def load_dataset(directory):

    image_paths=[]
    labels=[]
    for label in os.listdir(directory):
        # print(directory + "\\" + label)
        for filename in os.listdir(directory + "/" + label):
            img_path=os.path.join(directory,label,filename)
            image_paths.append(img_path)
            labels.append(label)
        print(label, 'Completed')
    return image_paths,labels

#  Convert data into Dataframe
train = pd.DataFrame()
train['image'],train['label'] = load_dataset(Train_Dir)
#  Shuffle the Dataset
train = train.sample(frac=1).reset_index(drop=True)
print(train)


from PIL import Image

train_features=[]
for img in range(len(train['image'])):
    imgs=Image.open(train['image'][img])
    # rescaleframe(imgs)
    imgs=np.array(imgs)
    # grey=cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)
    # canny=cv2.Canny(imgs,125,125)
    grey=cv2.resize(imgs,(48,48),interpolation=cv2.INTER_AREA)
    train_features.append(grey)
plt.imshow(grey,cmap='gray')
plt.show()

# Normalise the Image
x_train = np.array([i.astype('float32')/255.0 for i in train_features])
# print(x_train)
# Convert label to integer

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(train['label'])
y_train=le.transform(train['label'])

y_train=to_categorical(y_train,num_classes=7)
input_shape =(48,48,1)
# output_class=7
output_class=7

print(y_train[0])


#-----------------  MODEL CREATION --------------------#

model=Sequential()
#  Convolutional Layers
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(1024,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

# Fully Connected Layers
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.3))
#  Output Layer
model.add(Dense(output_class,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

#---------------TRAIN THE MODEL---------------#
history=model.fit(x=x_train,y=y_train,batch_size=512,epochs=300)
model.save('AJ_EDM_Mk6.h5')
