import numpy as np

def load_data():

    X_train = np.load('data/X_train.npy').astype(np.float32)
    y_train = np.load('data/y_train.npy').astype(np.int32)
    X_test = np.load('data/X_test.npy').astype(np.float32)
    y_test = np.load('data/y_test.npy').astype(np.int32)

    return (X_train,y_train),(X_test,y_test)

def read_random_pattern():
    X = np.load('data/X_rp.npy').astype(np.float32)
    y = np.load('data/y_rp.npy').astype(np.int32)
    return X,y

if __name__ == '__main__':
    
    # (X_train,y_train),(X_test,y_test) = load_data()

    from helpers import check_array_prop
    X,y = read_random_pattern()
    check_array_prop(X)
    check_array_prop(y)
