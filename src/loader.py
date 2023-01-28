import scipy.io

def get_data_XY():
    mat =  scipy.io.loadmat('./data/train_32x32.mat')
    X = mat['X']
    Y = mat['y']
    return X, Y

x, y = get_data_XY()
print(x.shape, y.shape)
