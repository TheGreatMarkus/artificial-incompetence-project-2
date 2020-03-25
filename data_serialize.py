# -----------------------------------------------------------
# main.py 2020-03-19
#
# File serialization 
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

import os

import pandas as pd

from constants import *
from ngram import Ngram


def data_ser_exists(v: int, n: int, delta: float) -> bool:
    """
    checks if appropriate vocab/ngram combination files exist
    :param v: Vocabulary for the model
    :param n: Ngram size for the model
    :param delta: Delta value for the model
    :return: True if files for all languages of appropriate combination exists, false otherwise.
    """
    for lang in LANGUAGES:
        if not (os.path.exists(TRAINING_FILE_TEMPLATE.format(lang, v, n, delta))):
            return False
    return True


def data_ser_load(v: int, n: int, delta: float):
    """
    loads the Ngram object, initializing DataFrames for each languages from proper files.
    :param v: Vocabulary for the model
    :param n: Ngram size for the model
    :param delta: Delta value for the model
    :return ngrams: Ngram object.
    """
    ngrams = Ngram(n)
    for lang in LANGUAGES:
        ngrams.ngrams[lang] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(lang, v, n, delta))
    return ngrams


def data_ser_save(ngrams: Ngram, v: int, n: int, delta: float):
    """
    stores each languages datafram to a file.
    :param ngrams: Ngram object containing all language dataFrames.
    :param v: Vocabulary for the model
    :param n: Ngram size for the model
    :param delta: Delta value for the model
    :reutn: void.
    """
    if not os.path.exists(TRAINING_RESULT_FOLDER):
        os.makedirs(TRAINING_RESULT_FOLDER)
    for lang in LANGUAGES:
        ngrams.ngrams[lang].to_pickle(TRAINING_FILE_TEMPLATE.format(lang, v, n, delta))
