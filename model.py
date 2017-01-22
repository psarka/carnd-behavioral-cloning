import itertools
import random

from keras.layers import Convolution2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
import numpy as np
import pandas as pd
from skimage import color
from skimage import transform
from skimage import io
from skimage import img_as_ubyte


DRIVING_LOG = pd.read_csv('udacity_data/driving_log.csv')
DRIVING_LOG = DRIVING_LOG.rename(columns={'center': 'center_image',
                                          'steering': 'steering_angle'})
DRIVING_LOG.center_image = 'udacity_data/' + DRIVING_LOG.center_image

N_TRAIN = len(DRIVING_LOG) * 8 // 10
N_VALIDATION = len(DRIVING_LOG) - N_TRAIN
IMAGE_SHAPE = (40, 40, 1)
BATCH_SIZE = 50


def preprocessed(image):


    #cut = image[70:110, :]
    cut = image[60:, :]
    scaled = transform.resize(cut, (40, 40))
    #return img_as_ubyte(scaled)

    gray = color.rgb2gray(scaled)
    norm = gray - np.mean(gray)
    return norm[:, :, None]


def training_stream():
    training_indices = list(DRIVING_LOG.index[:N_TRAIN])
    random.shuffle(training_indices)
    yield from batched(infinite_stream(training_indices))


def validation_stream():
    validation_indices = list(DRIVING_LOG.index[N_TRAIN:])
    yield from batched(infinite_stream(validation_indices))


def batched(iterable):
    while True:
        xs_ys = tuple(zip(*itertools.islice(iterable, BATCH_SIZE)))
        if not xs_ys:
            return
        yield np.stack(xs_ys[0]), np.stack(xs_ys[1])


def infinite_stream(indices):
    while True:
        yield from finite_stream(indices)


def finite_stream(indices):
    for i in indices:
        image_path = DRIVING_LOG.center_image[i]
        image = io.imread(image_path)
        steering_angle = DRIVING_LOG.steering_angle[i]

        yield preprocessed(image), steering_angle

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, 3, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile('adam', 'mean_squared_error')

if __name__ == '__main__':

    model.fit_generator(training_stream(),
                        samples_per_epoch=N_TRAIN,
                        validation_data=validation_stream(),
                        nb_val_samples=N_VALIDATION,
                        nb_epoch=5,
                        )

    model.save_weights('model1.h5')
    print('weights saved to model1.h5')
    with open('model1.json', 'w') as f:
        model = f.write(model.to_json())
        print('model saved to model1.json')
