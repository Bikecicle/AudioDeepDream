from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import numpy as np
import librosa


# variables
path = 'C:/Users/Griffin/Music/freesound-audio-tagging/'
sample_rate = 22050
grain_duration = 0.01

grain_size = int(sample_rate * grain_duration)

G = np.load('data/x_train_granular.npy')
G = G.reshape(G.shape[0], G.shape[1], 1)
input_grain = Input(shape=(grain_size, 1))  # adapt this if using `channels_first` image data format

x = Conv1D(16, 3, activation='relu', padding='same')(input_grain)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x)

x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_grain, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.summary()

autoencoder.fit(G, G,
                epochs=100,
                batch_size=1024,
                shuffle=True,
                validation_split=0.2)

