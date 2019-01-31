import numpy as np
from helpers import write_imgs_serially
from helpers import clear_folder
from helpers import normalize
from helpers import split_train_test

from explore_data import read_data, read_random_pattern_data

INPUT_SIZE = 96
OUTPUT_SIZE = 36

seed = 42


def prep_train_test():
    X, y = read_data()
    X = normalize(X, 0, 255)
    y = normalize(y, 0, 255)

    y = y > 127
    y = y * 1.0
    X /= 255.0

    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    print(X_train.shape)
    np.save('data/X_train.npy', X_train)
    print(y_train.shape)
    np.save('data/y_train.npy', y_train)
    print(X_test.shape)
    np.save('data/X_test.npy', X_test)
    print(y_test.shape)
    np.save('data/y_test.npy', y_test)

def prep_random_data():
    X, y = read_random_pattern_data()
    X = normalize(X, 0, 255)
    y = normalize(y, 0, 255)

    y = y > 127
    y = y * 1.0
    X /= 255.0

    print(X.shape)
    np.save('data/X_rp.npy', X)
    print(y.shape)
    np.save('data/y_rp.npy', y)

def see_imgs():
    # X_train = np.load('data/X_train.npy')
    # print(X_train.shape)
    # y_train = np.load('data/y_train.npy')
    # print(y_train.shape)
    # X_test = np.load('data/X_test.npy')
    # print(X_test.shape)
    # y_test = np.load('data/y_test.npy')
    # print(y_test.shape)

    X = np.load('data/X_rp.npy')
    print(X.shape)
    y = np.load('data/y_rp.npy')
    print(y.shape)

    X = normalize(X,0,255)
    y = y*255.0
    write_imgs_serially(X,base_path='tmp/sanity_in/')
    write_imgs_serially(y,base_path='tmp/sanity_out/')

if __name__ == '__main__':
    prep_train_test()
    prep_random_data()
    see_imgs()
