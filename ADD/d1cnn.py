from keras import Sequential
from keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
import json
import numpy as np
import os


def build_model():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    with open(os.path.join(config['paths']['data'], 'train_' + str(config['version']) + '.json'), 'r') as vconfig_file:
        vconfig = json.load(vconfig_file)

    sample_rate = vconfig['data_format']['sr']
    duration = vconfig['data_format']['dur']

    time_steps = int(sample_rate * duration)

    # load class list
    classes = vconfig['classes']
    num_classes = len(classes)
    print('Number of classes:', num_classes)

    print('Building model')
    model = Sequential()
    model.add(Reshape((time_steps, 1), input_shape=(time_steps,)))
    model.add(Conv1D(100, 10, activation='relu', input_shape=(time_steps, 1)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    print('Compiling model')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.save(os.path.join(config['paths']['models'], '1dcnn_' + str(config['version']) + '.h5'))
    return model

