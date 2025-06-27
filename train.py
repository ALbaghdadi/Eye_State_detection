import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from processing import append_closed,append_open,augment_image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping


path = r'C:\Users\hp\Desktop\eye open close\dataset\train_dataset'
closedLeftEyes = path + '\closedLeftEyes'
openLeftEyes =path + '\openLeftEyes'
closedRightEyes = path + '\closedRightEyes'
openRightEyes = path + '\openRightEyes'
close = r'C:\Users\hp\Desktop\eye open close\dataset\closed'
open  = r'C:\Users\hp\Desktop\eye open close\dataset\open'


open = append_open(open)
open_left = append_open(openLeftEyes)
open_right = append_open(openRightEyes)
close = append_closed(close)
closed_left=append_closed(closedLeftEyes)
closed_right = append_closed(closedRightEyes)


open_eyes= open_right 
closed_eyes= closed_right

open_labels = [ 1 for i in range(len(open_eyes)) ]
close_labels = [ 0 for i in range(len(closed_eyes)) ]
data = open_eyes+closed_eyes
x = np.array(data)/255 #Scaling
data2= open_labels+close_labels
y= np.array(data2)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(24, 24, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation='relu'),
   # layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test,y_test), callbacks=[early])

model.save("model.keras")