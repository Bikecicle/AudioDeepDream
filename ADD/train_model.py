from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import json
import os
from d1cnn import build_model

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


x_train = np.load(os.path.join(config['paths']['data'], 'x_train_' + str(config['version']) + '.npy'))
y_train = np.load(os.path.join(config['paths']['data'], 'y_train_' + str(config['version']) + '.npy'))


cp_callback = ModelCheckpoint(filepath=os.path.join(vconfig['paths']['weights'],
                                                    '1dcnn_' + str(config['version']),
                                                    'co.ckpt'),
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
