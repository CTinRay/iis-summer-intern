import pandas as pd
import numpy as np
from IPython import embed
import re
import pickle
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import random
import sys
import pdb, traceback, sys, ipdb
from tqdm import tqdm
from torch import LongTensor, FloatTensor
import math
import argparse

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('train', type=str, help='train', default='Train.csv')
parser.add_argument('embedding', type=str, help='wv.pkl', default='wv.pkl')
parser.add_argument('--test', action='append', help='Test',
                    default=['Develop.csv', 'Test.csv'])
parser.add_argument('--valid_ratio', type=float,
                    help='ratio of validation data.', default=0.1)
parser.add_argument('--n_iters', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--reg', type=float, default=1e-3,
                    help='regularization factor')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
args = parser.parse_args()

char_based = True
epochs = args.n_iters

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Tensor = FloatTensor

print ("Loading Dataset")
#  datasetName = ['Training Set', 'Development Set', 'Testing Set']
#  datasetPath = ['Train.csv', 'Develop.csv', 'Test.csv']
datasetName = [n.split('.')[0][:5] for n in [args.train] + args.test]
datasetPath = [args.train] + args.test
wordvectorPath = args.embedding
dataset = [pd.read_csv(path) for path in datasetPath]
train = 0

lbar_format = "{desc:>7s}{postfix} {percentage:3.0f}%"
rbar_format = "{n_fmt:^5s}/{total_fmt:^5s} [{elapsed}<{remaining}] {rate_fmt:>11s}"
bar_format = lbar_format + " |{bar}| " + rbar_format

class Preprocessor(object):
    def fit(self, data, *args, **kwargs):
        with open(wordvectorPath, 'rb') as f:
            wv = pickle.load(f)
            self.words = wv['word_dict']
        self.maxb = max(map(len, data['Body']))
        self.maxq = max(map(len, data['Question']))
        print ("Max body len: %d, Max query len: %d" % (self.maxb, self.maxq))

    def _preprocess(self, x):
        r = [[c
                for c in s
                if u'\u4e00' <= c <= u'\u9fff' and
                    c in self.words]
                for s in x]
        return r

    def _vectorize(self, x, l):
        r = [[self.words[c] for c in s] for s in x]
        r = [ s[:l] + [0] * max(0, l - len(s)) for s in r ]
        return r

    def transform(self, data):
        tb = self._preprocess(data['Body'].values)
        tq = self._preprocess(data['Question'].values)
        xb = self._vectorize(tb, self.maxb)
        xq = self._vectorize(tq, self.maxq)
        t = list(zip(tb, tq))
        x = list(zip(xb, xq))
        y = data['Operand'].values
        return x, y, t

    def __call__(self, *a, **k):
        self.transform(*a, **k)

preprocessor = Preprocessor()
preprocessor.fit(dataset[train])
dataset = [ preprocessor.transform(data) for data in dataset ]

class ResConv1d(nn.Module):
    """1D Convolution with Residual Connection."""
    def __init__(self, channels, kernel):
        super(ResConv1d, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, 1, (kernel-1)//2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform(m.weight)

    def forward(self, x):
        y = self.conv(x)
        y = (y + x) / math.sqrt(2)
        return y

class CNNBlock(nn.Module):
    """Convolution Layer for sentence."""
    def __init__(self):
        super(CNNBlock, self).__init__()
        with open('wv.pkl', 'rb') as f:
            wv = pickle.load(f)
            e = wv['embedding']
            # Set embedding of padding to zero
            e[0] = 0
            self.emb = nn.Embedding(*e.shape)
            self.emb.weight.data.copy_(torch.from_numpy(e))
            self.emb.weight.requires_grad = False
        self.conv = nn.Sequential(
            ResConv1d(300, 5),
            #  nn.Dropout(0.1),
            nn.LeakyReLU(),
            ResConv1d(300, 3),
            #  nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform(m.weight)

    def forward(self, x):
        x = torch.transpose(self.emb(x), 1, 2)
        x = self.conv(x)
        # Get max output
        x = F.max_pool1d(x, x.size()[-1]).squeeze(-1)
        return x

    def inpGrad(self, x):
        grad = self.emb(x).detach()
        grad.requires_grad = True
        x = torch.transpose(grad, 1, 2)
        x = self.conv(x)
        x = F.max_pool1d(x, x.size()[-1]).squeeze(-1)
        return x, grad

class CNNNet(nn.Module):
    def __init__(self, out):
        super(CNNNet, self).__init__()
        self.convb = CNNBlock()
        self.convq = CNNBlock()
        self.out = nn.Sequential(
            nn.Linear(600, 1000),
            #  nn.Dropout(0.2),
            nn.Linear(1000, out),
            #  nn.Dropout(0.2),
            nn.Softmax(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)

    def forward(self, b, q):
        b = self.convb(b)
        q = self.convq(q)
        x = torch.cat([b, q], dim=-1)
        #  x = b * q
        x = self.out(x)
        return x

    def inpGrad(self, b, q):
        b, gb = self.convb.inpGrad(b)
        q, gq = self.convq.inpGrad(q)
        x = torch.cat([b, q], dim=-1)
        #  x = b * q
        x = self.out(x)
        return x, (gb, gq)


class NNClassifier(object):
    def __init__(self, batchsize=200):
        self.batchsize = batchsize
        self.operators = None

    def fit(self, x, y, data):
        b, q = zip(*x)
        b, q = np.array(b), np.array(q)
        y = self._toIndex(y)

        # Split Validation Set
        ind = np.arange(len(x))
        np.random.shuffle(ind)
        valid = ind[:int(len(ind) * args.valid_ratio)]
        train = ind[int(len(ind) * args.valid_ratio):]

        # Create Network
        self.net = CNNNet(len(self.operators)).cuda()
        parameters = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=1e-5)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(epochs):  
            if epoch == 10:
                """Set embedding to trainable after CNN warmup"""
                print ("Trainable Embedding")
                optimizer.zero_grad()
                for m in self.net.modules():
                    if isinstance(m, nn.Embedding):
                        m.weight.requires_grad = True
                parameters = [p for p in self.net.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.reg)

            # Training Loop
            print ("Epoch: %d" % (epoch + 1))
            self.net.train()
            bar = tqdm(range(0, len(train), self.batchsize), smoothing=0,
                    desc="Optim", bar_format=bar_format)
            for step, i in enumerate(bar):
                i = train[i: i + self.batchsize]
                bb = Variable(torch.from_numpy(b[i]).cuda())
                bq = Variable(torch.from_numpy(q[i]).cuda())
                by = Variable(torch.from_numpy(y[i]).cuda())
                bp = self.net(bb, bq)
                
                loss = loss_function(bp, by)
                bar.postfix = "Loss = %10.6lf" % (loss.data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            np.random.shuffle(train)
            if data is not None:
                for test, name in data:
                    self.score(*test[:2], desc=name[:5])
            print ("")
            #  print (next(self.net.parameters()).grad.data)

    def predict(self, x, desc="Predi"):
        b, q = zip(*x)
        b, q = np.array(b), np.array(q)
        ind = np.arange(len(x))
        self.net.eval()
        ret = []
        bar = tqdm(range(0, len(x), self.batchsize), smoothing=0,
                desc=desc, bar_format=bar_format)
        bar.postfix = "                 "
        for step, i in enumerate(bar):
            i = ind[i: i + self.batchsize]
            predict = self._predict(b, q, i)
            ret += [self.opstr[o] for o in predict.cpu().numpy()]
        return ret

    def predict_proba(self, x, desc="Proba"):
        b, q = zip(*x)
        b, q = np.array(b), np.array(q)
        ind = np.arange(len(x))
        self.net.eval()
        ret = []
        bar = tqdm(range(0, len(x), self.batchsize), smoothing=0,
                desc=desc, bar_format=bar_format)
        bar.postfix = "                 "
        for step, i in enumerate(bar):
            i = ind[i: i + self.batchsize]
            predict = self._predict_proba(b, q, i).data
            ret += [[(self.opstr[o], p) for o, p in enumerate(y)] for y in predict.cpu().numpy()]
        return ret

    def score(self, x, y, desc="Score"):
        b, q = zip(*x)
        b, q = np.array(b), np.array(q)
        y = self._toIndex(y)
        ind = np.arange(len(x))
        return self._score(b, q, y, ind, desc)

    def _score(self, b, q, y, data, desc="Score"):
        correct = 0
        size = 0
        self.net.eval()
        bar = tqdm(range(0, len(data), self.batchsize), smoothing=0,
                desc=desc, bar_format=bar_format)
        for step, i in enumerate(bar):
            i = data[i: i + self.batchsize]
            predict = self._predict(b, q, i)
            label = torch.from_numpy(y[i]).cuda()
            correct += sum(predict == label)
            size += label.size(0)
            bar.postfix = "Accu = %10.6lf" % (correct / size)
        return correct / size

    def _predict(self, *args, **kwargs):
        proba = self._predict_proba(*args, **kwargs)
        res = torch.max(proba, 1)[1].data.squeeze()
        return res

    def _predict_proba(self, b, q, i, volatile=True):
        self.net.eval()
        bb = Variable(torch.from_numpy(b[i]).cuda(), volatile=volatile)
        bq = Variable(torch.from_numpy(q[i]).cuda(), volatile=volatile)
        bp = self.net(bb, bq)
        return bp

    def grad(self, x, y):
        b, q = x
        self.net.eval()
        self.net.zero_grad()
        bb = Variable(torch.LongTensor(b).cuda()).unsqueeze(0)
        bq = Variable(torch.LongTensor(q).cuda()).unsqueeze(0)
        by = torch.zeros(1, len(self.operators)).cuda()
        by[0][self.operators[y]] = 1
        bp, g = self.net.inpGrad(bb, bq)
        bp.backward(by)
        g = [gg.grad.data.squeeze(0).cpu() for gg in g]
        return g

    def _toIndex(self, y):
        if not self.operators:
            self.operators = {}
            self.opstr = np.unique(y)
            for i, o in enumerate(self.opstr):
                self.operators[o] = i
        ind = [self.operators[o] for o in y]
        return np.array(ind)

def predict(body, query):
    df = pd.DataFrame()
    df['Body'] = [body]
    df['Question'] = [query]
    df['Operand'] = ['+']
    data = preprocessor.transform(df)
    for o in model.opstr:
        print("Importance of operator %s" % o)
        grad = model.grad(data[0][0], o)
        for g, t in zip(grad, data[2][0]):
            g = torch.abs(g).sum(-1)
            g = torch.abs((g - g.mean()) / g.std())
            pprint([ "%s  %.4lf" % p for p in zip(t, g)])
        print ("")
    print("Probability")
    print(model.predict_proba(data[0])[0])
try:
    model = NNClassifier(args.batch_size)
    data = list(zip(dataset, datasetName))
    try:
        model.fit(dataset[train][0], dataset[train][1], data)
    except KeyboardInterrupt:
        pass

    labels = np.unique(dataset[train][1])

    for data, name in zip(dataset, datasetName):
        print (name)
        print ("Accuracy: %.4lf" % (model.score(data[0], data[1])))

        py = np.array(model.predict(data[0]))
        cm = confusion_matrix(py, data[1], labels)

        print("CM, normalized by predict")
        cmnorm = cm * 100 / (cm.sum(axis = 1).reshape(-1, 1))
        cmnorm[np.logical_not(np.isfinite(cmnorm))] = 0
        print (("%3s "*(len(labels)+1)) % (('p\\t',) + tuple(labels)))
        print ('\n'.join([("%3s " % l) + (("%3d "*len(p)) % tuple(p)) for l, p in zip(labels, cmnorm)]))

        print("\nCM, normalized by true label")
        cmnorm = cm * 100 / (cm.sum(axis = 0))
        cmnorm[np.logical_not(np.isfinite(cmnorm))] = 0
        print (("%3s "*(len(labels)+1)) % (('p\\t',) + tuple(labels)))
        print ('\n'.join([("%3s " % l) + (("%3d "*len(p)) % tuple(p)) for l, p in zip(labels, cmnorm)]))
        print ("\n")

    print ("Gradient of input 1 (e.g importance of each words)")
    for i in range(1):
        grad = model.grad(dataset[train][0][i], dataset[train][1][i])
        for g, t in zip(grad, dataset[train][2][i]):
            g = torch.abs(g).sum(-1)
            g = torch.abs((g - g.mean()) / g.std())
            pprint([ "%s  %.4lf" % p for p in zip(t, g)])
        print ("")
except KeyboardInterrupt:
    pass
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
embed()
