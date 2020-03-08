# -----------------------------------------------------------
# vocabulary.py 2020-03-08
#
# Define functions for generating models based on training data
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
from abc import abstractmethod

import pandas as pd
from collections import Counter
from constants import DF_COLUMN_TWEET, UNIGRAM, BIGRAM, TRIGRAM
from utils import get_frequency_of_tokens
from custom_tokenize import tokenize_unigram


class Ngram:
    """
    The Ngram defines the class of concrete ngram actions
    """

    def __init__(self, ngram_option: str):
        """
        Init strategy
        """
        if ngram_option == UNIGRAM:
            self.ngram = Unigram()
        elif ngram_option == BIGRAM:
            self.ngram = Bigram()
        elif ngram_option == TRIGRAM:
            self.ngram = Trigram()
        else:
            raise ValueError('Invalid ngram option!')

    def generate(self, df):
        """
        Generate ngram according to the strategy
        """
        return self.ngram.do_generate(df)


class NgramStrategy:
    """
    Interface for all ngrams
    """

    @abstractmethod
    def do_generate(self, df):
        pass


#########################
# unigram
#########################
class Unigram(NgramStrategy):
    def do_generate(self, df):
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
class Bigram(NgramStrategy):
    def do_generate(self, df):
        """
        Generate
        :param df:
        :return:
        """
        pass


#########################
# trigram
#########################
class Trigram(NgramStrategy):
    def do_generate(self, df):
        """
        Generate
        :param df:
        :return:
        """
        pass
