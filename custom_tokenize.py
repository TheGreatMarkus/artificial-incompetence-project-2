from typing import List

from constants import OUT_OF_VOCABULARY_DELIM


def tokenize(tweet: str, span: int) -> List[str]:
    """
    Tokenize tweets by a single legal character
    :param span: Execution range. Ex: Unigram = 1; Bigram = 2; Trigram = 3
    :param tweet: Tweet input
    :return: List of tokens
    """
    tweet_len = len(tweet)
    return [tweet[c:c + span] for c in range(tweet_len) if (c + span) <= tweet_len and
            OUT_OF_VOCABULARY_DELIM not in tweet[c:c + span]]
