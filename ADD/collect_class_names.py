import librosa
import json


# variables
path = 'C:/Users/Griffin/Music/freesound-audio-tagging/'
sample_rate = 11025
duration = 1

x_train = [[]]
y_train = []


i = 0
for line in open(path + 'train.csv', 'r'):
    vals = line.split(',')
    '''x_raw, sr = librosa.load(path + 'audio_train/' + vals[0], sr=sample_rate, duration=duration)
    x = []
    for xi in x_raw:
        x.append(float(xi))
    x_train.append(x)'''
    y_train.append(vals[1])
    print(i)
    i += 1

classes = []

for label in y_train:
    if label not in classes:
        classes.append(label)
num_classes = len(classes)

data = {'classes': classes}
with open('classes.json', 'w') as out:
    json.dump(data, out)
