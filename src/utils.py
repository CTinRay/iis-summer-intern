import pdb
import numpy as np
import pandas as pd


def encode_text(text, word_dict):
    encoded = []
    for row in text:
        words = row.strip().split()
        word_indices = []
        for word in words:
            if word in word_dict:
                word_indices.append(word_dict[word])
            else:
                # [TODO] Cope with OOV
                pass

        encoded.append(word_indices)

    return encoded


def pad_text(text, max_len=None):
    if max_len is None:
        max_len = max(map(len, text))

    padded = np.zeros((len(text), max_len))

    for i in range(len(text)):
        to_copy = min(len(text[i]), max_len)
        padded[i, :to_copy] = text[i][:to_copy]

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


def encode_labels(labels):
    operators = ['+', '-', '*', '/', '//', '%']
    indices = [operators.index(label) for label in labels]
    encoded = np.zeros((len(labels), len(operators)))
    encoded[np.arange(len(labels)), indices] = 1
    return encoded


def BatchGenerator(X, y, batch_size, shuffle=True):
    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

    for i in range(X.shape[0] // batch_size + 1):
        batch = {'x': X[i * batch_size: (i + 1) * batch_size],
                 'y': y[i * batch_size: (i + 1) * batch_size]}
        yield batch
