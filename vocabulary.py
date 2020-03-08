# -----------------------------------------------------------
# vocabulary.py 2020-03-08
#
# Define functions for transforming raw list of tweets to fit vocabulary
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
from abc import abstractmethod
import re

from constants import VOCABULARY_0, VOCABULARY_1, VOCABULARY_2, DF_COLUMN_TWEET


class Vocabulary:
    """
    The Vocabulary defines the class of concrete transformations
    """

    def __init__(self, transformation: str):
        """
        Init strategy
        """
        if transformation == VOCABULARY_0:
            self.transformation = TransformToV0()
        elif transformation == VOCABULARY_1:
            self.transformation = TransformToV1()
        elif transformation == VOCABULARY_2:
            self.transformation = TransformToV2()
        else:
            raise ValueError('Invalid vocabulary processing option!')

    def transform(self, df):
        """
        Transform tweets according to the strategy
        """
        self.transformation.do_transform(df)


class TransformStrategy:
    """
    Interface for all transformations
    """

    @abstractmethod
    def do_transform(self, df):
        pass


#########################
# V0
#########################
class TransformToV0(TransformStrategy):
    def do_transform(self, df):
        """
            TODO
            :param df:
            :return:
            """
        pass


#########################
# V1
#########################
class TransformToV1(TransformStrategy):
    def do_transform(self, df):
        """
            Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
            :param df: Input DataFrame
            :return: void
            """
        df[DF_COLUMN_TWEET] = df[DF_COLUMN_TWEET].map(lambda tweet: re.sub('[^a-zA-Z]', '*', tweet))


#########################
# V2
#########################
class TransformToV2(TransformStrategy):
    def do_transform(self, df):
        """
            TODO
            :param df:
            :return:
            """
        pass
