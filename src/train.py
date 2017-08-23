import argparse
import numpy as np
from nn import NNClassifier
from utils import Preprocessor, Interactive
import pdb
import sys
import traceback
import pandas as pd
from IPython import embed
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
    parser.add_argument('--batch_size', type=int, default=256,
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
                       embedding=preprocessor.embedding, early_stop=30)
    clf.fit(train['x'], train['y'])

    valid['y_'], valid['y_prob'] = clf.predict(valid['x'], True)
    train['y_'], train['y_prob'] = clf.predict(train['x'], True, True)

    operators = ['+','-','*','/','%', 's']    

    for i in range(len(operators)):
        indices = np.where(valid['y'][:, i] == 1)
        count = np.sum(valid['y_'][indices], axis=0)
        print('operator %s: %d' % (operators[i], len(indices[0])))
        for j in range(len(operators)):
            print('    %s: %d' % (operators[j], count[j]))

    data = pd.read_csv(args.valid)
    data["Predict"] = list(valid['y_'])
    data["Predict"] = data["Predict"].map(lambda x:operators[list(x).index(1)])
    #print(data.shape)
    data["Prob"] = list(valid["y_prob"])
    data = data[data["Predict"] != data["Operand"]]
    #print(data.shape)
    data.to_csv("incorrect.csv")

    data = pd.read_csv(args.train)
    
    data["Predict"] = list(train['y_'])
    data["Predict"] = data["Predict"].map(lambda x:operators[list(x).index(1)])
    #print(data.shape)
    data["Prob"] = list(train["y_prob"])
    data = data[data["Operand"]!="s"]
    data = data[data["Predict"] != data["Operand"]]
    #print(data.shape)
    data.to_csv("incorrect_train.csv")
    # interactive interface
    while True:
        Interactive(clf.predict, preprocessor, operators)

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
#        pdb.post_mortem(tb)
