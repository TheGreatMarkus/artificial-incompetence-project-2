from typing import List

from constants import OUT_OF_VOCABULARY_DELIM


def tokenize(tweet: str, span: int) -> List[str]:
    """
    Tokenize tweets given a length for the tokens.

    :param span: Execution range. Ex: Unigram = 1 character; Bigram = 2 characters; Trigram = 3 characters
    :param tweet: Tweet input string
    :return: List of tokens
    """
    tweet_len = len(tweet)
    return [tweet[c:c + span] for c in range(tweet_len) if (c + span) <= tweet_len and
            OUT_OF_VOCABULARY_DELIM not in tweet[c:c + span]]
