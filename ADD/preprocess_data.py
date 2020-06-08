from keras.utils import to_categorical
import json
import numpy
import librosa

# variables
path = 'C:/Users/Griffin/Music/freesound-audio-tagging/'
sample_rate = 11025
duration = 0.1

time_steps = int(sample_rate * duration)

x_train = []
y_train_raw = []

# load class list
with open('classes.json', 'r') as class_file:
    classes = json.load(class_file)['classes']
num_classes = len(classes)
print('Number of classes:', num_classes)

# load segments
files = 0
samples = 0
for line in open(path + 'train.csv', 'r'):
    vals = line.split(',')
    if vals[1] in classes:
        xs_raw, sr = librosa.load(path + 'audio_train/' + vals[0], sr=sample_rate)
        t = 0
        while t <= len(xs_raw):
            xs = []
            for i in range(time_steps):
                if t >= len(xs_raw):
                    xs.append(0)
                else:
                    xs.append(xs_raw[t])
                t += 1
            x_train.append(xs)
            y_train_raw.append(vals[1])
            print('File', str(files), "- sample", str(samples), 'loaded:', vals[1])
            samples += 1
        files += 1

y_train = []
for label in y_train_raw:
    y_train.append(classes.index(label))

x_train = numpy.asarray(x_train)
y_train = to_categorical(y_train, num_classes=num_classes)
print(x_train.shape, y_train.shape)

