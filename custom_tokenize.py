from constants import IGNORE_DELIM


def tokenize_unigram(tweet: str):
    """
    Tokenize tweet by a single legal character
    :param tweet: Tweet input
    :return: List of tokens
    """
    return [c for c in tweet if c != IGNORE_DELIM]
