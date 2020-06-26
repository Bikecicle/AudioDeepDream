from keras import Sequential
from keras.layers import Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
import json
import numpy as np
import os


def build_model():
    sample_rate = 22050
    duration = 1

    time_steps = int(sample_rate * duration)

    # load class list
    classes = [
        "Cello",
        "Double_bass",
        "Electric_piano"
    ]
    num_classes = len(classes)
    print('Number of classes:', num_classes)

    print('Building model')
    model = Sequential()
    model.add(Reshape((time_steps, 1), input_shape=(time_steps,)))
    model.add(Conv2D(100, 10, activation='relu', input_shape=(time_steps, 1)))
    model.add(Conv2D(100, 10, activation='relu'))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(160, 10, activation='relu'))
    model.add(Conv2D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    print('Compiling model')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.save('models/2dcnn.h5')
    return model

