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

TRAINING_FILE_TEMPLATE = 'trainingResults/{}{}_{}.pkl'

def ifExists(vocab: int, ngram: int):
    """
    checks if appropriate vocab/ngram combination files exist
    :param vocab: which vocabulary is being addressed.
    :param ngram: which ngram is being addressed.
    :return: True if files for all languages of appropriate combination exists, false otherwise.
    """
    if(os.path.exists(TRAINING_FILE_TEMPLATE.format(LANG_EU, vocab, ngram))
    and os.path.exists(TRAINING_FILE_TEMPLATE.format(LANG_CA, vocab, ngram))
    and os.path.exists(TRAINING_FILE_TEMPLATE.format(LANG_GL, vocab, ngram))
    and os.path.exists(TRAINING_FILE_TEMPLATE.format(LANG_ES, vocab, ngram))
    and os.path.exists(TRAINING_FILE_TEMPLATE.format(LANG_EN, vocab, ngram))
    and os.path.exists(TRAINING_FILE_TEMPLATE.format(LANG_PT, vocab, ngram))):
        return True
    else:
        return False
    

def loadNgrams(vocab: int, ngram: int):
    """
    loads the Ngram object, initializing DataFrames for each languages from proper files.
    :param vocab: which vocabulary is being addressed.
    :param ngram: which ngram is being addressed.
    :return ngrams: Ngram object.
    """
    ngrams =Ngram(ngram)
    ngrams.ngrams[LANG_EU] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(LANG_EU, vocab, ngram))
    ngrams.ngrams[LANG_CA] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(LANG_CA, vocab, ngram))
    ngrams.ngrams[LANG_GL] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(LANG_GL, vocab, ngram))
    ngrams.ngrams[LANG_ES] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(LANG_ES, vocab, ngram))
    ngrams.ngrams[LANG_EN] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(LANG_EN, vocab, ngram))
    ngrams.ngrams[LANG_PT] = pd.read_pickle(TRAINING_FILE_TEMPLATE.format(LANG_PT, vocab, ngram))
    return ngrams


def saveNgrams(ngrams: Ngram, vocab: int, ngram: int):
    """
    stores each languages datafram to a file.
    :param ngrams: Ngram object containing all language dataFrames.
    :param vocab: which vocabulary was used. Needed for proper naming.
    :param ngram: which ngram was used. Needed for proper naming.
    :reutn: void.
    """
    ngrams.ngrams[LANG_EU].to_pickle(TRAINING_FILE_TEMPLATE.format(LANG_EU, vocab, ngram))
    ngrams.ngrams[LANG_CA].to_pickle(TRAINING_FILE_TEMPLATE.format(LANG_CA, vocab, ngram))
    ngrams.ngrams[LANG_GL].to_pickle(TRAINING_FILE_TEMPLATE.format(LANG_GL, vocab, ngram))
    ngrams.ngrams[LANG_ES].to_pickle(TRAINING_FILE_TEMPLATE.format(LANG_ES, vocab, ngram))
    ngrams.ngrams[LANG_EN].to_pickle(TRAINING_FILE_TEMPLATE.format(LANG_EN, vocab, ngram))
    ngrams.ngrams[LANG_PT].to_pickle(TRAINING_FILE_TEMPLATE.format(LANG_PT, vocab, ngram))