import numpy as np
from helpers import write_imgs_serially
from helpers import clear_folder
from helpers import normalize

INPUT_SIZE = 96
OUTPUT_SIZE = 36

seed = 42


def read_data():
    X = np.load('data/x_MNIST.p.npy')
    y = np.load('data/y_MNIST.p.npy')
    print(X.shape)
    print(y.shape)
    return X, y


def read_random_pattern_data():
    X = np.load('data/random_pattern/x_rand_pattern.npy')
    y = np.load('data/random_pattern/y_rand_pattern.npy')
    print(X.shape)
    print(y.shape)
    return X, y


def see_imgs():
    X, y = read_data()
    X = normalize(X, 0, 255)
    y = normalize(y, 0, 255)
    X = X.reshape(-1, INPUT_SIZE, INPUT_SIZE)
    write_imgs_serially(X, base_path='tmp/all/input/')
    y = y.reshape(-1, OUTPUT_SIZE, OUTPUT_SIZE)
    write_imgs_serially(y, base_path='tmp/all/output/')
    del X, y

    X, y = read_random_pattern_data()
    X = normalize(X, 0, 255)
    y = normalize(y, 0, 255)
    X = X.reshape(-1, INPUT_SIZE, INPUT_SIZE)
    write_imgs_serially(X, base_path='tmp/all/rp_input/')
    y = y.reshape(-1, OUTPUT_SIZE, OUTPUT_SIZE)
    write_imgs_serially(y, base_path='tmp/all/rp_output/')


if __name__ == '__main__':
    see_imgs()
    # read_random_pattern_data()
