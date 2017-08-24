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
        description='Convert mwp.uwds.ilds.cmds_complete.xlsx'
                    ' to our input format')
    parser.add_argument('input', type=str, help='The xlsx filename')
    parser.add_argument('output', type=str, help='Output csv directory')
    args = parser.parse_args()

    for sheet in ['ai2', 'ilds', 'cmds']:
        df = pd.read_excel(args.input, sheetname=sheet,
                           header=0, index_col=0)
        parse_xlsx(df)
        df.to_csv(os.path.join(args.output, '%s.csv' % sheet))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
