from keras import Sequential
from keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from keras.utils import to_categorical
import json
import numpy
import librosa



'''
print('Building model')
model_m = Sequential()
model_m.add(Reshape((time_steps, 1), input_shape=(time_steps,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(time_steps, 1)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())

print('Compiling model')
model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

print('Training model')
model_m.fit(x_train, y_train, batch_size=10, epochs=50, validation_split=0.2, verbose=1)

model_m.save('1dcnn.h5')
'''


