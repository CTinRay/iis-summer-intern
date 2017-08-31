# Dynamic Coattention Model

## Description
Implement the first part of dynamic coattention model. 

## Reference
https://arxiv.org/abs/1611.01604

## Prerequisite
'''
pip3 install requirements.txt
'''
Note that the version of tensorflow package is 1.3.0, not 0.12 or so.

## How to run
```
python3 train.py [train_set] [valid_set] [word_vector_embedding]
```

where

- train set is a csv file.
- valid set is a csv file.
- word vector embedding is a pickle made by make_embedding.py

The program will save model to directory `checkpoint/`, and the training process will be save to directory with name specified by `--name` argument.
