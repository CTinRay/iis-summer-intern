import argparse
import numpy as np
from nn import NNClassifier
from utils import Preprocessor
import pdb
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('valid', type=str, help='valid.csv')
    parser.add_argument('embedding', type=str, help='embedding.pickle')
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
    parser.add_argument('--name', type=str, default='dnn',
                        help='name for the model, will be the directory name for summary')
    args = parser.parse_args()

    preprocessor = Preprocessor(args.embedding)
    train = preprocessor.load_data(args.train)
    valid = preprocessor.load_data(args.valid)

    clf = NNClassifier(valid=valid,
                       learning_rate=args.lr,
                       n_iters=args.n_iters,
                       name=args.name,
                       batch_size=args.batch_size,
                       embedding=preprocessor.embedding)
    clf.fit(train['x'], train['y'])
    clf.predict(train['x'])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
