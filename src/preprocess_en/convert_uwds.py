"""
Convert mwp.uwds.ilds.cmds_complete.xlsx to our input format
"""
import re
import os
import argparse
import pandas as pd
import pdb
import sys
import traceback


def parse_ai2(df):
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
        last_punctuate_index = punctuate_matches[-1].span()[0]
        questions.append(question[:last_punctuate_index + 1])
        bodies.append(question[last_punctuate_index + 1:])
    df.drop('question', axis=1, inplace=True)
    df['Body'] = bodies
    df['Question'] = questions


def parse_ilds(input, output):
    df = pd.read_excel(input, header=0, sheetname='ilds')
    pass


def cmds(input, output):
    df = pd.read_excel(input, header=0, sheetname='cmds')
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Convert mwp.uwds.ilds.cmds_complete.xlsx'
                    ' to our input format')
    parser.add_argument('input', type=str, help='The xlsx filename')
    parser.add_argument('output', type=str, help='Output csv directory')
    args = parser.parse_args()

    df_ai2 = pd.read_excel(args.input, sheetname='ai2',
                           header=0, index_col=0)
    parse_ai2(df_ai2)
    df_ai2.to_csv(os.path.join(args.output, 'ai2.csv'))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
