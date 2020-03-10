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
from vocabulary import transform_to_v1
from ngram import Ngram


def main(v: int, n: int, delta: float, train_file: str, test_file: str):
    train_data = pd.read_csv(train_file, delimiter='\t', names=[DF_COLUMN_ID, DF_COLUMN_NAME,
                                                                DF_COLUMN_LANG, DF_COLUMN_TWEET])
    print('(Rows, Columns) => {}'.format(train_data.shape))
    transform_to_v1(train_data)
    ngrams = Ngram(n)
    ngrams.generate(train_data)

    print(ngrams.print_ngrams())


main(VOCABULARY_1, UNIGRAM, 0.5, './training-tweets.txt', './test-tweets.txt')
