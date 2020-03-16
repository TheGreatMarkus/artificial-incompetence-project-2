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

    def generate(self, df):
        """
        Generate ngram counts from tweets
        :param df: DataFrame of raw data.
        :return: void
        """
        for language, tweets in df.groupby(DF_COLUMN_LANG)[DF_COLUMN_TWEET]:
            tokenized_tweets = tweets.map(lambda tweet: tokenize(tweet, self.ngram_option))
            token_count = tokenized_tweets.map(lambda tweet: Counter(tweet))
            token_count_sum = sum(token_count, Counter())
            for token, count in token_count_sum.items():
                if len(token) > 1:
                    self.ngrams[language].loc[token[:-1], token[-1:]] = count
                else:
                    self.ngrams[language].insert(0, token, [count], True)

        self.__finalize_ngrams()

    def print_ngrams(self):
        """
        Print ngrams for each language.
        :return:
        """
        for lang, df in self.ngrams.items():
            print('\nNgram for {} language'.format(lang))
            print('{}\n'.format(df))

    def __finalize_ngrams(self):
        """
        Insert 0.0 in missing cells.
        Generate sum column which holds sums for each row.
        :return:
        """
        for ngram in self.ngrams.values():
            ngram.fillna(0.0, inplace=True)
            ngram[DF_COLUMN_SUM] = ngram.sum(axis=1)
