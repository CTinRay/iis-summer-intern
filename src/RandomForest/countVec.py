import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.ensemble import *
from sklearn.metrics import *
from pprint import pprint

char_based = True

print ("Loading Dataset")
datasetName = ['Training Set', 'Development Set', 'Testing Set']
datasetPath = ['Train.csv', 'Develop.csv', 'Test.csv']
dataset = [pd.read_csv(path) for path in datasetPath]
train = 0

def preprocess(x):
    # Filter out non-chinese char
    if char_based:
        corpus = [c for c in x if u'\u4e00' <= c <= u'\u9fff']
        return ''.join(corpus)
    else:
        x = x.split(u'\u3000')
        corpus = [w for w in x if all(u'\u4e00' <= c <= u'\u9fff' for c in w)]
        return ' '.join(corpus)

print ("Initialize Vectorizer")
corpus = np.array([str(u'\u3000'.join(p)) for p in zip(dataset[train]['Body'].values, dataset[train]['Question'].values)])

ops = dataset[train]['Operand'].values
optype = np.unique(ops)
corpus = ['\u3000'.join(corpus[ops == op]) for op in optype]

if char_based:
    vectorizer = CountVectorizer(preprocessor=preprocess, token_pattern=r'.')
else:
    vectorizer = CountVectorizer(preprocessor=preprocess)

vectorizer.fit(corpus)


print ("Vectorize data")
def vectorize(data):
    body = vectorizer.transform(data['Body']).toarray()
    query = vectorizer.transform(data['Question']).toarray()
    x = np.concatenate([body, query], axis=1)
    y = data['Operand'].values
    return (x, y)
dataset = list(map(vectorize, dataset))

print ("Fitting Model")
model = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)

model.fit(dataset[train][0], dataset[train][1])
print ("\n")

#  pprint (list(zip(m.predict(develx), devely)))
labels = np.unique(dataset[train][1])

for data, name in zip(dataset, datasetName):
    print (name)
    print ("Accuracy: %.4lf" % (model.score(data[0], data[1])))

    py = model.predict(data[0])
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

print("Feature importance")
fn = vectorizer.get_feature_names()
fi = model.feature_importances_
f = list(sorted(zip(fn, fi), key=lambda x: -x[1]))[:20]
pprint(f)
