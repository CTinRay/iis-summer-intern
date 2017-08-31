import pandas as pd
import os
import argparse
import sys
import traceback
import pdb
import numpy as np
from nn import NNClassifier
from utils import Preprocessor

def k_fold_cross_validation(df, prefix, args):
    # Parse index
    index_list = parse(args.k_fold_index_filename)

    # Index to k fold dataframe
    k_df_list = index_to_dataframe(index_list, df)

    # Write DataFrame to csv file
    write_to_csv(k_df_list, prefix)

    # Cross validation 
    preprocessor = Preprocessor(args.embedding)
    K = len(args.k_fold_index_filename)
    metric = 0.0
    assert K == 5
    for i in range(0, K):
        # Initialize
        train = preprocessor.load_data(prefix + "train" + str(i) + ".csv")
        valid = preprocessor.load_data(prefix + "valid" + str(i) + ".csv")
        # Train
        clf = NNClassifier(valid=valid,
                       learning_rate=args.lr,
                       n_iters=args.n_iters,
                       name=args.name,
                       batch_size=args.batch_size,
                       embedding=preprocessor.embedding,
                       early_stop=20)
        clf.fit(train['x'], train['y'])
        # Evaluate
        valid['y_'] = clf.predict(valid['x'])
        metric += accuracy(valid['y_'], valid['y'])

    return metric / 5


def accuracy(y_, y):
    acc = (np.argmax(y_, axis=1) == np.argmax(y, axis=1))
    acc = np.sum(acc) / acc.size
    return acc

def read_data(filename):
    df = pd.read_csv(filename, header=0)
    return df


def write_to_csv(k_df_list, prefix):
    for i, valid_df in enumerate(k_df_list):
        valid_df.to_csv(prefix + "valid"+ str(i) + ".csv", index=False)
        train_df_list = []
        for j, train_df in enumerate(k_df_list):
            if i != j:
                train_df_list.append(train_df)
        train_df_list = pd.concat(train_df_list)
        train_df_list.to_csv(prefix + "train" + str(i) + ".csv", index=False)

def parse(k_fold_index_filename):
    '''
    Return a list of k fold indices
    ex. [[1, 6, 9, 10] , [10, 15, 16]]
    '''
    index_list = []
    for k, filename in enumerate(k_fold_index_filename):
        with open(filename) as f:
            local_list = []
            for row in f:
                local_list.append(int(row.split('-')[1]))
            index_list.append(local_list)

    return index_list


def index_to_dataframe(k_fold_index_list, df):
    '''
    Return a fold with panda DataFrame
    '''
    k_df_list = []
    for k, index_list in enumerate(k_fold_index_list):
        data = {'Body': [], 'Question': [], 'Operand': []}
        for i in index_list:
            # Index is 0-based
            data['Body'].append(df['Body'][i-1])
            data['Question'].append(df['Question'][i-1])
            data['Operand'].append(df['Operand'][i-1])
        k_df = pd.DataFrame(data=data)
        k_df_list.append(k_df)

    return k_df_list

def main():
    parser = argparse.ArgumentParser(description="Run ilds dataset 5-fold cross validation")
    parser.add_argument('ildspath', type=str, help="ilds dataset path")
    parser.add_argument('embedding', type=str, help='embedding.pickle')
    parser.add_argument('k_fold_index_filename', type=str, help="id list path", nargs=5)
    parser.add_argument('--n_iters', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--name', type=str, default='dnn', 
        help='name for the model, will be the directory name for summary')
    args = parser.parse_args()
    
    # Read data
    df = read_data(args.ildspath)

    # K fold cross validatin
    metric = k_fold_cross_validation(df, "ilds", args)

    print('5 fold cross validation result: {} '.format(metric))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
