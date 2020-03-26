# -----------------------------------------------------------
# constants.py 2020-03-25
#
# Define constants for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
import os
import pandas as pd
import data_serialize as ds
from vocabulary import transform_to_vocab
from ngram import Ngram
from constants import *


def generate_trace_file(v: int, n: int, delta: float, result_df: pd.DataFrame):
    """
    Generate trace file.
    :param v: Vocabulary choice
    :param n: ngram choice
    :param delta: Smoothing choice
    :param result_df: result DataFrame
    :return: void
    """
    if not os.path.exists(TRACE_FILE_DIR):
        os.makedirs(TRACE_FILE_DIR)
    with open(TRACE_FILE_TEMPLATE.format(v, n, delta), 'w') as f:
        result_df.to_string(f, index=False, col_space=OUTPUT_FILE_SPACE_COUNT)


def process_train_data(v: int, n: int, delta: float, vocab_size: int, train_file: str) -> Ngram:
    """
    Wrapper function for the training data processing.
    Either fetch or generate necessary Ngrams based on the training information.
    :param v: Vocabulary choice
    :param n: ngram choice
    :param delta: Smoothing choice
    :param vocab_size: The size of the vocabulary
    :param train_file: Path to training data
    :return: Ngram
    """
    ngrams = Ngram(n)
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
        ngrams.generate(train_data, delta, vocab_size)
        ds.data_ser_save(ngrams, v, n, delta)
    return ngrams

def validate_params(v: int, n: int, delta: float, train_file: str, test_file: str):
    if v not in VALID_VOCABULARIES:
        raise ValueError(VOCABULARY_VALUE_ERROR_MESSAGE)
    if n not in VALID_NGRAMS:
        raise ValueError(NGRAM_VALUE_ERROR_MESSAGE)
    if delta <= 0 or delta > 1:
        raise ValueError(DELTA_VALUE_ERROR_MESSAGE)
    if not os.path.exists(train_file):
        raise ValueError(MISSING_TRAIN_FILE_ERROR_MESSAGE)
    if not os.path.exists(test_file):
        raise ValueError(MISSING_TEST_FILE_ERROR_MESSAGE)
