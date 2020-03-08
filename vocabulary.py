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
from constants import Tweet, DF_COLUMN_TWEET


#########################
# V0
#########################
def transform_to_v0(tweets: List[Tweet], v: int) -> List[Tweet]:
    """
    TODO
    :param tweets:
    :param v:
    :return:
    """
    pass


#########################
# V1
#########################
def transform_to_v1(df):
    """
    Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
    :param df: Input DataFrame
    :return: void
    """
    df[DF_COLUMN_TWEET] = df[DF_COLUMN_TWEET].map(lambda tweet: re.sub('[^a-zA-Z]', '*', tweet))


#########################
# V2
#########################
def transform_to_v2(tweets: List[Tweet], v: int) -> List[Tweet]:
    """
    TODO
    :param tweets:
    :param v:
    :return:
    """
    pass
