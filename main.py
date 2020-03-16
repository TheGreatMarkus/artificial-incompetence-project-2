# -----------------------------------------------------------
# main.py 2020-03-08
#
# Define entry point for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
import pandas as pd

from constants import *
from ngram import Ngram
from vocabulary import transform_to_v1, transform_to_v0, transform_to_v2


def main(v: int, n: int, delta: float, train_file: str, test_file: str):
    """
    Entry point of program.
    :param v: Vocabulary choice
    :param n: ngram choice
    :param delta: Smoothing choice
    :param train_file: Path to training data
    :param test_file: Path to testing data
    :return: void
    """
    train_data = pd.read_csv(train_file,
                             delimiter='\t',
                             names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])

    print('Input Training Data (Rows, Columns) => {}'.format(train_data.shape))

    if v == VOCABULARY_0:
        transform_to_v0(train_data)
    elif v == VOCABULARY_1:
        transform_to_v1(train_data)
    elif v == VOCABULARY_2:
        transform_to_v2(train_data)

    ngrams = Ngram(n)
    ngrams.generate(train_data)

    print(ngrams.print_ngrams())


main(VOCABULARY_0, UNIGRAM, 0.5, './training-tweets.txt', './test-tweets.txt')
