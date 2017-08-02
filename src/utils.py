import numpy as np


def BatchGenerator(X, y, batch_size, shuffle=True):
    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

    for i in range(X.shape[0] // batch_size + 1):
        batch = {'x': X[i * batch_size: (i + 1) * batch_size],
                 'y': y[i * batch_size: (i + 1) * batch_size]}
        yield batch
