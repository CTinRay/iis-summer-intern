"""
Convert nlunds.json to our input format
"""
import re
import os
import argparse
import json
import pandas as pd
import pdb
import sys
import traceback


def parse_xlsx(df):
    operands = []
    for formula in df['formula']:
        ops = list(filter(lambda x: x in '+-*', formula))
        ops = sorted(ops)
        ops = ''.join(ops)
        operands.append(ops)
    df['Operand'] = operands

    bodies = []
    questions = []
    for question in df['question']:
        # split body and question
        punctuate_matches = list(re.finditer('[,.]', question))
        if len(punctuate_matches) > 0:
            last_punctuate_index = punctuate_matches[-1].span()[0]
        else:
            last_punctuate_index = -1
        bodies.append(question[:last_punctuate_index + 1])
        questions.append(question[last_punctuate_index + 1:])
    df.drop('question', axis=1, inplace=True)
    df['Body'] = bodies
    df['Question'] = questions


def main():
    parser = argparse.ArgumentParser(
        description='Convert nlunds.json'
                    ' to our input format')
    parser.add_argument('input', type=str, help='The json filename')
    parser.add_argument('output', type=str, help='Output csv filename')
    args = parser.parse_args()

    with open(args.input) as f:
        problems = json.load(f)

    ids = []
    operators = []
    answers = []
    questions = []
    bodies = []
    for problem in problems:
        ids.append(problem[0])

        # extract operator type
        ops = list(filter(lambda x: x in '+-*/', problem[1]))
        ops = sorted(ops)
        ops = ''.join(ops)
        operators.append(ops)

        answers.append(problem[2])

        # split body and question
        punctuate_matches = list(re.finditer('[,.]', problem[4]))
        if len(punctuate_matches) > 0:
            last_punctuate_index = punctuate_matches[-1].span()[0]
        else:
            last_punctuate_index = -1
        bodies.append(problem[4][:last_punctuate_index + 1])
        questions.append(problem[4][last_punctuate_index + 1:])

    df = pd.DataFrame({'Operand': operators,
                       'Answer': answers,
                       'Question': questions,
                       'Body': bodies}, index=ids)
    df.to_csv(args.output)



        

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
