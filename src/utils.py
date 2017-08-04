import pdb
import numpy as np
import pandas as pd


def encode_text(text, word_dict):
    encoded = []
    for row in text:
        words = row.strip().split(' ')
        word_indices = [word_dict[word] for word in words]
        encoded.append(word_indices)

    return encoded


def pad_text(text, max_len=None):
    if max_len is None:
        max_len = max(map(len, text))

    padded = np.zeros((len(text), max_len))

    for i in range(len(text)):
        padded[i] = text[i][:max_len]

    return padded


def split_valid(data, valid_ratio, shuffle=True):
    indices = np.arange(data['x'].shape[0])
    if shuffle:
        np.random.shuffle(indices)
        data['x'] = data['x'][indices]
        data['y'] = data['y'][indices]

    n_valid = int(data['x'].shape[0] * valid_ratio)
    train = {'x': data['x'][n_valid:], 'y': data['y'][n_valid:]}
    valid = {'x': data['x'][:n_valid], 'y': data['y'][:n_valid]}

    return train, valid, indices


def BatchGenerator(X, y, batch_size, shuffle=True):
    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

    for i in range(X.shape[0] // batch_size + 1):
        batch = {'x': X[i * batch_size: (i + 1) * batch_size],
                 'y': y[i * batch_size: (i + 1) * batch_size]}
        yield batch
