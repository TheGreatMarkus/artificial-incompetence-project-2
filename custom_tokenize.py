from typing import List

from constants import OUT_OF_VOCABULARY_DELIM, LEFT_PAD_SYMBOL, RIGHT_PAD_SYMBOL


def tokenize(tweet: str, span: int, extended_func: bool = False,
             left_pad_symbol: str = LEFT_PAD_SYMBOL, right_pad_symbol: str = RIGHT_PAD_SYMBOL) -> List[str]:
    """
    Tokenize tweets given a length for the tokens.

    :param left_pad_symbol: Padding symbol for the tweet start
    :param right_pad_symbol: Padding symbol for the tweet end
    :param span: Execution range. Ex: Unigram = 1 character; Bigram = 2 characters; Trigram = 3 characters
    :param tweet: Tweet input string
    :param extended_func: If True, apply extended functionality for BYOM
    :return: List of tokens
    """
    if extended_func:
        tweet = left_pad_symbol * (span - 1) + tweet + right_pad_symbol * (span - 1)
    tweet_len = len(tweet)
    return [tweet[c:c + span] for c in range(tweet_len) if (c + span) <= tweet_len and
            OUT_OF_VOCABULARY_DELIM not in tweet[c:c + span]]
