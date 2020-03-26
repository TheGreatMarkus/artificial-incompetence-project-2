# -----------------------------------------------------------
# vocabulary.py 2020-03-08
#
# Define functions for generating models based on training data
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
from collections import Counter

import pandas as pd

from constants import *
from custom_tokenize import tokenize


class Ngram:
    """
    Wrapper class for Ngrams manipulation.
    """

    def __init__(self, ngram_option: int):
        """
        Init languages with empty DataFrames and a choice for Ngram
        """
        self.ngrams = {LANG_EU: pd.DataFrame(), LANG_CA: pd.DataFrame(), LANG_GL: pd.DataFrame(),
                       LANG_ES: pd.DataFrame(), LANG_EN: pd.DataFrame(), LANG_PT: pd.DataFrame()}
        self.ngram_option = ngram_option

    def generate(self, df, delta: float, vocab_size: int):
        """
        Generate ngram counts from tweets
        :param df: DataFrame of raw data.
        :param delta: Delta value for the model.
        :param vocab_size: Size of the vocabulary
        :return: void
        """
        for language, tweets in df.groupby(DF_COLUMN_LANG)[DF_COLUMN_TWEET]:
            tokenized_tweets = tweets.map(lambda tweet: tokenize(tweet, self.ngram_option))
            token_count = tokenized_tweets.map(lambda tweet: Counter(tweet))
            token_count_sum = sum(token_count, Counter())
            for token, count in token_count_sum.items():
                self.ngrams[language].loc[token[:-1], token[-1:]] = count

        self.__finalize_ngrams()
        for lang, df in self.ngrams.items():
            row_sum = self.ngrams[lang].sum(axis=1)
            self.ngrams[lang] = (df + delta).div(row_sum + vocab_size * delta, axis=0)
            self.ngrams[lang][DF_COLUMN_OOV] = delta / (row_sum + vocab_size * delta)

    def print(self):
        """
        Print ngrams for each language.
        :return:
        """
        for lang, df in self.ngrams.items():
            print('\nNgram for the {} language'.format(lang))
            print('{}\n'.format(df))

    def __finalize_ngrams(self):
        """
        Insert 0.0 in missing cells.
        :return:
        """
        for lang in LANGUAGES:
            self.ngrams[lang].fillna(0.0, inplace=True)
