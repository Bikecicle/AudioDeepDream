import tensorflow as tf
from keras import backend as k
from d1cnn import build_model
import numpy as np
import librosa
import math

model = build_model()
model.load_weights('weights/1dcnn_0/co.ckpt')

def load_sound(filename):
    xs_raw, sr = librosa.load('inputs/test_in.wav', sr=11025)
    return np.asarray(xs_raw)


def save_sound(sound, filename):
    librosa.output.write_wav(filename, sound)


def get_clip_size(num_samples, clip_size=400):
    num_clips = int(round(num_samples/clip_size))
    num_clips = max(1, num_clips)
    actual_clip_size = math.ceil(num_samples/num_clips)
    return actual_clip_size

def optimize_sound(layer_tensor, sound, num_iterations=10, step_size=3.0, clip_size=400):
    sound = sound.copy()
    print('Processing sound: ')
    gradient =