import pickle
import numpy as np
import pandas as pd
import pdb
import os

class Preprocessor:

    @staticmethod
    def encode_text(text, word_dict):
        encoded = []
        for row in text:
            # Word-based
            #words = row.strip().split()
            # Char-based
            words = [c for c in row if u'\u4e00'<= c <= u'\u9fff']
            word_indices = []
            for word in words:
                if word in word_dict:
                    word_indices.append(word_dict[word])
                else:
                    # [TODO] Cope with OOV
                    pass

            encoded.append(word_indices)

        return encoded

    @staticmethod
    def pad_text(text, max_len=None):
        if max_len is None:
            max_len = max(map(len, text))
    
        padded = np.zeros((len(text), max_len))

        for i in range(len(text)):
            to_copy = min(len(text[i]), max_len)
            padded[i, :to_copy] = text[i][:to_copy]

        return padded

    @staticmethod
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

    @staticmethod
    def encode_labels(labels):
        operators = ['+','-','*','/','%']
        indices = [operators.index(label) for label in labels]
        encoded = np.zeros((len(labels), len(operators)))
        encoded[np.arange(len(labels)), indices] = 1
        return encoded

    def __init__(self, embedding_filename):

        # load embedding and make embedding matrix
        with open(embedding_filename, 'rb') as f:
            obj = pickle.load(f)
            self.word_dict = obj['word_dict']
            self.embedding = obj['embedding']

        self._max_len = None

    def load_data(self, filename):
        # read data
        df = pd.read_csv(filename)

        # preprocess text
        data = {}
        data['Body'] = self.encode_text(df['Body'],
                                        self.word_dict)
        data['Question'] = self.encode_text(df['Question'],
                                            self.word_dict)

        if self._max_len is None:
            max_len_body = max(map(len, data['Body']))
            max_len_question = max(map(len, data['Question']))
            self._max_len = max(max_len_body, max_len_question)

        data['Body'] = self.pad_text(data['Body'], self._max_len)
        data['Question'] = self.pad_text(data['Question'], self._max_len)
        # stack Body and Question as x
        data['x'] = np.zeros((data['Body'].shape[0],
                              2,
                              data['Body'].shape[1]))
        data['x'][:, 0, :] = data['Body']
        data['x'][:, 1, :data['Question'].shape[1]] = data['Question']

        # preprocess tags to one hot
        data['y'] = self.encode_labels(df['Operand'])

        return data


def BatchGenerator(X, y, batch_size, shuffle=True):
    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

    for i in range(X.shape[0] // batch_size + 1):
        batch = {'x': X[i * batch_size: (i + 1) * batch_size],
                 'y': y[i * batch_size: (i + 1) * batch_size]}
        yield batch
def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
def Interactive(predict_func, preprocessor, operators):
    test = {}
    test['Body'] = [input("Body: ")]
    test['Question'] = [input("Question: ")]
    test['Operand'] = [input("Operand: ")]
    # dump to temp file
    tempdf = pd.DataFrame(data=test)
    tempdf.to_csv("temp.csv")
    # load and preprocess
    test = preprocessor.load_data("temp.csv")
    test['y_'], test['y_prob'] = predict_func(test['x'], True)
    test['y_prob'] = test['y_prob'].reshape((-1))
    for i, op in enumerate(operators):
        print("Op:{}, predict prob: {}".format(op, test['y_prob'][i]))

