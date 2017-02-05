import itertools

from keras.layers import Convolution2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import transform
from skimage import io
from sklearn.utils import shuffle

FWD_DATASETS = [
    ('udacity_data', 0.99),  # normal driving data given by Udacity
    ('driving_0', 0.99),  # normal driving data collected additionally
    ('extra_turn_0', 0.01),  # extra data for the tricky corner
    ('extra_turn_1', 0.99),  # extra data for the tricky corner (for validation)
    # ('recovery_0', 1),  # data with snippets of recovery
]

SIDE_DATASETS = [
    'udacity_data'  # normal driving data given by Udacity, left & right cameras
]


#
# Data reading and streaming. Read the csv files, extract useful data and
# split into training and validation parts.
#

def train_collections():

    for dataset, proportion in FWD_DATASETS:

        log = pd.read_csv(dataset + '/driving_log.csv', header=None, skiprows=1)
        paths = dataset + '/IMG/' + log[0].str.split('/').str[-1]
        angles = log[3]

        yield paths[:int(len(paths)*proportion)], angles[:int(len(angles)*proportion)]

    for dataset in SIDE_DATASETS:

        log = pd.read_csv(dataset + '/driving_log.csv', header=None, skiprows=1)
        left_paths = dataset + '/IMG/' + log[1].str.split('/').str[-1]
        right_paths = dataset + '/IMG/' + log[2].str.split('/').str[-1]
        angles = log[3]

        yield left_paths, angles + 0.3
        yield right_paths, angles - 0.3


def validation_collections():

    for dataset, proportion in FWD_DATASETS:

        log = pd.read_csv(dataset + '/driving_log.csv', header=None, skiprows=1)
        paths = dataset + '/IMG/' + log[0].str.split('/').str[-1]
        angles = log[3]

        yield paths[int(len(paths)*proportion):], angles[int(len(angles)*proportion):]


def flat(collection):
    return [(path, angle) for paths_angles in collection for path, angle in zip(*paths_angles)]


train_paths_angles = shuffle(flat(train_collections()))
validation_paths_angles = flat(validation_collections())

N_TRAIN = len(train_paths_angles)
N_VALIDATION = len(validation_paths_angles)
BATCH_SIZE = 50


def image_angle_stream(paths_angles):
    while True:
        for image_path, angle in paths_angles:
            if Path(image_path).exists():
                yield io.imread(image_path), angle


def batched(iterable):
    while True:
        xs_ys = tuple(zip(*itertools.islice(iterable, BATCH_SIZE)))
        if not xs_ys:
            return
        yield np.stack(xs_ys[0]), np.stack(xs_ys[1])


#
# Preprocessing and augmentation
#


IMAGE_SHAPE = (40, 40, 3)


def preprocess(image):
    cut = image[60:, :]
    scaled = transform.resize(cut, (40, 40))

    norm = scaled - np.mean(scaled)
    norm /= np.std(norm)
    return norm


def preprocessed(image_angle_iterable):
    for image, angle in image_angle_iterable:
        yield preprocess(image), angle


def shift(image, k):

    if k > 0:
        shifted = image[:, k:, :]
        for i in range(k):
            shifted = np.append(shifted, np.roll(shifted[:, -1:, :], 1, axis=0), axis=1)

    else:
        shifted = image[:, :k, :]
        for i in range(-k):
            shifted = np.append(np.roll(shifted[:, -1:, :], 1, axis=0), shifted, axis=1)

    return shifted


def augmented(image_angle_iterable):
    for image, angle in image_angle_iterable:

        fimage, fangle = np.fliplr(image), -angle

        yield image, angle*2
        yield fimage, fangle*2
        # yield transform.rotate(image, 10, mode='wrap'), angle - 0.1
        # yield transform.rotate(fimage, 10, mode='wrap'), fangle - 0.1
        # yield transform.rotate(image, -10, mode='wrap'), angle + 0.1
        # yield transform.rotate(fimage, -10, mode='wrap'), fangle + 0.1
        # yield shift(image, 5), angle - 0.1
        # yield shift(fimage, 5), fangle - 0.1
        # yield shift(image, -5), angle + 0.1
        # yield shift(fimage, -5), fangle + 0.1


N_AUGMENTED = len(list(augmented([(np.zeros((20, 20, 3)), 0)])))


#
# Composed streams
#

def training_stream():
    yield from batched(augmented(preprocessed(image_angle_stream(train_paths_angles))))


def validation_stream():
    yield from batched(preprocessed(image_angle_stream(validation_paths_angles)))


#
# Deep conv net
#

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

    print(N_TRAIN)
    print(N_AUGMENTED)
    model.fit_generator(training_stream(),
                        samples_per_epoch=N_TRAIN * N_AUGMENTED,
                        validation_data=validation_stream(),
                        nb_val_samples=N_VALIDATION,
                        nb_epoch=2,
                        )

    model.save_weights('model.h5')
    with open('model1.json', 'w') as f:
        model = f.write(model.to_json())
