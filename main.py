# -----------------------------------------------------------
# main.py 2020-03-08
#
# Define entry point for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

import pandas as pd

import data_serialize as ds
from constants import *
from ngram import Ngram
from test_results import get_test_results
from vocabulary import transform_to_vocab, get_vocab_size


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
    test_data = pd.read_csv(test_file, delimiter='\t',
                            names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])
    transform_to_vocab(test_data, v)
    vocab_size = get_vocab_size(v)

    print("Creating model with parameters: [vocabulary = {}, ngram size = {}, delta = {}]".format(v, n, delta))

    if ds.data_ser_exists(v, n, delta):
        print("Model with parameters already stored. Retrieving")
        ngrams = ds.data_ser_load(v, n, delta)
    else:
        print("Model with parameters not stored. Generating model from provided training data")
        train_data = pd.read_csv(train_file,
                                 delimiter='\t',
                                 names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])
        transform_to_vocab(train_data, v)
        print("Shape of Training Data (Rows, Columns) => {}".format(train_data.shape))
        ngrams = Ngram(n)
        ngrams.generate(train_data, delta, vocab_size)
        ds.data_ser_save(ngrams, v, n, delta)

    ngrams.print()

    print("Running model against provided testing data.")
    results = get_test_results(test_data, ngrams, vocab_size, n)

    print("Final results generated")
    print(results)

    print("Evaluating classifier with parameters: [vocabulary = {}, ngram size = {}, delta = {}]".format(v, n, delta))
    total = len(results.index)
    correct = len(results.loc[results[DF_COLUMN_ACTUAL] == results[DF_COLUMN_GUESS]])
    print("Accuracy: {0:.4f}%".format(correct / total))


main(VOCABULARY_1, TRIGRAM, 1, './training-tweets.txt', './test-tweets.txt')
