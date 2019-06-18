import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import SHAPE, batch_gen
import argparse
import os


NAME = "very_simple_100-1000-100"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
np.random.seed(0)

data_dir = '/Users/julienkessels/Desktop/data/'
test_size = 0.2
keep_prob = 0.5
nb_epoch = 10
samples_per_epoch = 20000
batch_size = 40
learning_rate = 1.0e-4


def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    # devide between training and testing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid



def build_model_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=SHAPE))
    model.add(Conv2D(2, 3, 3, activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D((4,4),(4,4),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    return model

def train_model(model, X_train, X_valid, y_train, y_valid):
    # Saves the model after every epoch.
    checkpoint = ModelCheckpoint('modell-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    # MSE + adaptive gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    #Fit model with the generated, augmented, data in batches
    model.fit_generator(batch_gen(data_dir, X_train, y_train, batch_size, True),
                        samples_per_epoch,
                        nb_epoch,
                        max_q_size=1,
                        validation_data=batch_gen(data_dir, X_valid, y_valid, batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint, tensorboard],
                        verbose=1)


def main():
    data = load_data()
    model = build_model()
    train_model(model, *data)


if __name__ == '__main__':
    main()
