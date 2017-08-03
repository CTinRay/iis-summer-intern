import numpy as np
import argparse
from utils import load_embedding, encode_text, pad_text, split_valid
import pandas as pd
import pdb
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('embedding', type=str, help='embedding.txt')
    parser.add_argument('--preprocess_args', type=str,
                        default='preprocess_args.pickle',
                        help='pickle to store preprocess arguments')
    parser.add_argument('--valid_ratio', type=float,
                        help='ratio of validation data.', default=0.1)
    parser.add_argument('--n_iters', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    args = parser.parse_args()

    # read data
    train_df = pd.read_csv(args.train)

    # load embedding and make embedding matrix
    word_dict, embedding_matrix = load_embedding(args.embedding)

    # preprocess text
    train_data = {}
    train_data['Body'] = encode_text(train_df['Body'],
                                     word_dict)
    train_data['Body'] = pad_text(train_data['Body'])
    train_data['Question'] = encode_text(train_df['Question'],
                                         word_dict)
    train_data['Question'] = pad_text(train_data['Question'])

    # stack Body and Question as x
    train_data['x'] = np.zeros((train_data['Body'].shape[0],
                                2,
                                train_data['Body'].shape[1]))
    train_data['x'][:, 0, :] = train_data['Body']
    train_data['x'][:, 1, :train_data['Question'].shape[1]] = \
        train_data['Question']

    # split data
    train, valid, _ = split_valid(train_data, args.valid_ratio)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
