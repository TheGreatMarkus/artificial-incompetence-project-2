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
from vocabulary import Vocabulary
from ngram import Ngram


def main(v: str, n: str, delta: float, training_file: str, testing_file: str):
    train_data = pd.read_csv(training_file, delimiter='\t', names=[DF_COLUMN_ID, DF_COLUMN_NAME,
                                                                   DF_COLUMN_LANG, DF_COLUMN_TWEET])
    vocabulary = Vocabulary(v)
    ngram = Ngram(n)

    vocabulary.transform(train_data)
    ngram_df = ngram.generate(train_data)

    print(ngram_df)


main('V1', 'unigram', 1, './training-tweets.txt', '')
