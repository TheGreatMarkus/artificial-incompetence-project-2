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
from ngram import generate_unigram


def main():
    """
    TODO
    :return:
    """
    train_data = pd.read_csv('./training-tweets.txt', delimiter='\t', names=[DF_COLUMN_ID, DF_COLUMN_NAME,
                                                                             DF_COLUMN_LANG, DF_COLUMN_TWEET])
    transform_to_v2(train_data)
    unigram_df = generate_unigram(train_data)
    print(unigram_df)
main()
