# -----------------------------------------------------------
# main.py 2020-03-19
#
# File serialization 
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

import os.path

import pandas as pd

from constants import *
from ngram import Ngram


def ifExists(vocab: int, ngram: int):
    """
    checks if appropriate vocab/ngram combination files exist
    :param vocab: which vocabulary is being addressed.
    :param ngram: which ngram is being addressed.
    :return: True if files for all languages of appropriate combination exists, false otherwise.
    """
    for lang in LANGUAGES:
        if not (os.path.exists(TRAINING_FILE_TEMPLATE.format(lang, vocab, ngram))):
            return False
    return True


def loadNgrams(vocab: int, ngram: int):
    """
    loads the Ngram object, initializing DataFrames for each languages from proper files.
    :param vocab: which vocabulary is being addressed.
    :param ngram: which ngram is being addressed.
    :return ngrams: Ngram object.
    """
    ngrams = Ngram(ngram)
    for lang in LANGUAGES:
        ngrams.ngrams[lang] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(lang, vocab, ngram))
    return ngrams


def saveNgrams(ngrams: Ngram, vocab: int, ngram: int):
    """
    stores each languages datafram to a file.
    :param ngrams: Ngram object containing all language dataFrames.
    :param vocab: which vocabulary was used. Needed for proper naming.
    :param ngram: which ngram was used. Needed for proper naming.
    :reutn: void.
    """
    for lang in LANGUAGES:
        ngrams.ngrams[lang].to_pickle(TRAINING_FILE_TEMPLATE.format(lang, vocab, ngram))
