# -----------------------------------------------------------
# vocabulary.py 2020-03-08
#
# Define functions for transforming raw list of tweets to fit vocabulary
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
import re

from pandas import DataFrame

from constants import DF_COLUMN_TWEET, OUT_OF_VOCABULARY_DELIM, VOCABULARY_0, VOCABULARY_1, VOCABULARY_2, \
    VOCABULARY_0_SIZE, VOCABULARY_1_SIZE, VOCABULARY_2_SIZE


def transform_to_vocab(df: DataFrame, v: int) -> None:
    """
    Transform Tweets of a DataFrame to fit the given vocabulary.
    :param df: The DataFrame
    :param v: the vocabulary
    :return: void
    """
    if v == VOCABULARY_0:
        transform_to_v0(df)
    elif v == VOCABULARY_1:
        transform_to_v1(df)
    elif v == VOCABULARY_2:
        transform_to_v2(df)


def get_vocab_size(v: int) -> int:
    """
    Returns the size of a given vocabulary.
    :param v: The vocabulary number
    :return: The size of the vocabulary.
    """
    if v == VOCABULARY_0:
        return VOCABULARY_0_SIZE
    elif v == VOCABULARY_1:
        return VOCABULARY_1_SIZE
    elif v == VOCABULARY_2:
        return VOCABULARY_2_SIZE


def transform_to_v0(df: DataFrame) -> None:
    """
    Convert all tweet texts to lower case letters.

    Any non-letter character will be replaced with *.
    :param df: Input DataFrame
    :return: void
    """
    df[DF_COLUMN_TWEET] = df[DF_COLUMN_TWEET].map(
        lambda tweet: re.sub('[^a-zA-Z]', OUT_OF_VOCABULARY_DELIM, tweet).lower())


def transform_to_v1(df: DataFrame) -> None:
    """
    Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
    :param df: Input DataFrame
    :return: void
    """
    df[DF_COLUMN_TWEET] = df[DF_COLUMN_TWEET].map(lambda tweet: re.sub('[^a-zA-Z]', OUT_OF_VOCABULARY_DELIM, tweet))


def transform_to_v2(df: DataFrame) -> None:
    """
    Keeps all characters returned by isalpha(), all others replaced by '*'
    :param df: Dataframe structure containing all tweets.
    :return: void
    """
    df[DF_COLUMN_TWEET] = df[DF_COLUMN_TWEET].map(
        lambda tweet: ''.join([i if i.isalpha() else OUT_OF_VOCABULARY_DELIM for i in tweet]))
