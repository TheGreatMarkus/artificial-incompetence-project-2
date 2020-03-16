# -----------------------------------------------------------
# vocabulary.py 2020-03-08
#
# Define functions for transforming raw list of tweets to fit vocabulary
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
from typing import List
import re

from pandas import DataFrame

from constants import DF_COLUMN_TWEET, OUT_OF_VOCABULARY_DELIM


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


#########################
# V2
#########################
def transform_to_v2(df: DataFrame) -> None:
    """
    TODO
    :param df: Input DataFrame
    :return: void
    """
    pass
