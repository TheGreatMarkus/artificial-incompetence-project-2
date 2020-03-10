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
        if self.ngram_option == UNIGRAM:
            self.__generate_unigrams(df)
        elif self.ngram_option == BIGRAM:
            self.__generate_bigrams(df)
        elif self.ngram_option == TRIGRAM:
            self.__generate_trigrams(df)
        else:
            raise ValueError('Invalid ngram option!')
        self.__finalize_ngrams()

    def print_ngrams(self):
        for lang, df in self.ngrams.items():
            print('\nNgram for {} language'.format(lang))
            print('{}\n'.format(df))

    def __generate_unigrams(self, df):
        """
        Generate Bag of Words unigrams for each language label
        :param df: Input DataFrame
        :return: void
        """
        for language, tweets in df.groupby(DF_COLUMN_LANG)[DF_COLUMN_TWEET]:  # Groupby language, select only Tweet column
            tokenized_tweets = tweets.map(lambda tweet: tokenize(tweet, self.ngram_option))  # Split each tweet in a row into tokens
            char_frequencies = tokenized_tweets.map(lambda tweet: Counter(tweet))  # Count frequency of each token in row
            sum_char_frequencies = sum(char_frequencies, Counter())  # Accumulate all Counters along each row
            self.ngrams[language] = pd.DataFrame([sum_char_frequencies])  # Convert total Counter object to a DataFrame

    def __generate_bigrams(self, df):
        """
        TODO
        :param tweets:
        :param v:
        :param n:
        :param delta:
        :return: 2D array
        """
        pass

    def __generate_trigrams(self, df):
        """
        TODO
        :param tweets:
        :param v:
        :param n:
        :param delta:
        :return: 3D array
        """
        pass

    def __finalize_ngrams(self):
        """
        Insert 0.0 in missing cells.
        Generate sum column which holds sums for each row.
        :return:
        """
        for ngram in self.ngrams.values():
            ngram.fillna(0.0, inplace=True)
            ngram[DF_COLUMN_SUM] = ngram.sum(axis=1)
