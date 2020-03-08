# -----------------------------------------------------------
# vocabulary.py 2020-03-08
#
# Define functions for generating models based on training data
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

import pandas as pd
from typing import List
from collections import Counter
from constants import Tweet, DF_COLUMN_TWEET
from utils import get_frequency_of_tokens
from tokenize import tokenize_unigram

#########################
# unigram
#########################
def generate_unigram(df):
    """
    Generate Bag of Words unigram
    :param df: Input
    :return: DataFrame of frequencies
    """
    frequencies = [get_frequency_of_tokens(tokenize_unigram(tweet)) for tweet in df[DF_COLUMN_TWEET]]
    counter = sum(frequencies, Counter())
    unigram_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    unigram_df = unigram_df.rename(columns={'index': 'Character', 0: 'Frequency'})
    return unigram_df


#########################
# bigram
#########################
def generate_bigram(tweets: List[Tweet], v: int, n: int, delta: float):
    """
    TODO
    :param tweets:
    :param v:
    :param n:
    :param delta:
    :return: 2D array
    """
    pass


#########################
# trigram
#########################
def generate_trigram(tweets: List[Tweet], v: int, n: int, delta: float):
    """
    TODO
    :param tweets:
    :param v:
    :param n:
    :param delta:
    :return: 3D array
    """
    pass
