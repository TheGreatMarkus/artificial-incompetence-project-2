from collections import Counter
from typing import List


def get_frequency_of_tokens(tokens: List[str]):
    """
    Get frequency of tokens
    :param tokens: List of tokens
    :return: Counter of frequencies for each token
    """
    return Counter(tokens)
