from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from d2cnn import build_model

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

model = build_model()
model.fit(x_train,
          y_train,
          batch_size=100,
          epochs=5,
          validation_split=0.2,
          verbose=1,
          callbacks=[cp_callback])
