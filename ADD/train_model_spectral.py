from keras import Sequential
from keras.layers import Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

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


x_train = np.load('data/x_train_spectral.npy')
y_train = np.load('data/y_train_spectral.npy')


cp_callback = ModelCheckpoint(filepath=os.path.join('weights/d2cnn/co.ckpt'),
                              save_weights_only=True,
                              verbose=1)
print(x_train.shape)
print('Building model')
model = Sequential()
model.add(Reshape((1025, 44, 1), input_shape=(1025, 44)))
model.add(Conv2D(100, 4, activation='relu', input_shape=(1025, 44, 1)))
model.add(Conv2D(100, 4, activation='relu'))
model.add(MaxPooling2D(3))
model.add(Conv2D(160, 4, activation='relu'))
model.add(Conv2D(160, 4, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

print('Compiling model')
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=100,
          epochs=5,
          validation_split=0.2,
          verbose=1,
          callbacks=[cp_callback])
