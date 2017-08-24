"""
Convert embedding as pickle that contains a dictionary:
- word_dict (dict): dictionary that map word to its index in the embedding.
- embedding (numpy matrix): embedding matrix of words.
"""
import argparse
import numpy as np
import pickle
import pdb
import sys
import traceback


def load_embedding(filename):
    word_dict = {}
    embedding = []

    with open(filename, errors='ignore') as f:
        index = 0

        # ignore first row
        next(f)
        for row in f:
            cols = row[1:].rstrip().split(' ')
            vec = np.asarray(cols[1:], dtype='float32')
            embedding.append(vec)
            word = row[0] + cols[0]
            word_dict[word] = index
            index += 1
            
    embedding = np.array(embedding)
    return word_dict, embedding


def main():
    parser = argparse.ArgumentParser(description='Convert embedding to pickle')
    parser.add_argument('input', type=str, help='embedding.txt')
    parser.add_argument('output', type=str, help='embedding.pickle')
    args = parser.parse_args()

    word_dict, embedding = load_embedding(args.input)

    with open(args.output, 'wb') as f:
        pickle.dump({'word_dict': word_dict,
                     'embedding': embedding}, f)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
