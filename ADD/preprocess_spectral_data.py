from keras.utils import to_categorical
import json
import numpy as np
import librosa
import os

# variables
path = 'C:/Users/Griffin/Music/freesound-audio-tagging/'
sample_rate = 22050
duration = 1

time_steps = int(sample_rate * duration)

x_train = []
y_train_raw = []

# load class list
classes = [
    "Cello",
    "Double_bass",
    "Electric_piano"
]
num_classes = len(classes)
print('Number of classes:', num_classes)

# load segments
files = 0
for line in open(path + 'train.csv', 'r'):
    vals = line.split(',')
    if vals[1] in classes:
        xs, sr = librosa.load(path + 'audio_train/' + vals[0], sr=sample_rate, duration=duration)
        xs_padded = np.zeros(time_steps)
        xs_padded[:len(xs)] = xs
        xs_spec = np.abs(librosa.stft(xs_padded))
        x_train.append(xs_spec)
        y_train_raw.append(vals[1])
        print('File', str(files), 'loaded:', vals[1])
        files += 1

y_train = []
for label in y_train_raw:
    y_train.append(classes.index(label))

x_train = np.asarray(x_train)
y_train = to_categorical(np.array(y_train), num_classes=num_classes)
print(x_train.shape, y_train.shape)

np.save(os.path.join('data', 'x_train_spectral.npy'), x_train)
np.save(os.path.join('data', 'y_train_spectral.npy'), y_train)
