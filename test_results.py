from math import log10

import pandas as pd
from pandas import DataFrame

from constants import DF_COLUMN_ID, DF_COLUMN_GUESS, DF_COLUMN_SCORE, DF_COLUMN_ACTUAL, DF_COLUMN_LANG, LANGUAGES, \
    DF_COLUMN_OOV, DF_COLUMN_TWEET
from custom_tokenize import tokenize
from ngram import Ngram


def get_test_results(test_data: DataFrame, ngrams: Ngram, vocab_size: int, n: int) -> DataFrame:
    """
    Generate test result DataFrame from test data and model
    :param test_data: The test data to evaluate
    :param ngrams: The models
    :param vocab_size: The size of the vocabulary
    :param n: The size of the ngrams
    :return: Result DataFrame
    """
    results = pd.DataFrame(columns=[DF_COLUMN_ID, DF_COLUMN_GUESS, DF_COLUMN_SCORE, DF_COLUMN_ACTUAL], dtype=float)
    results[DF_COLUMN_ID] = test_data[DF_COLUMN_ID]
    results[DF_COLUMN_ACTUAL] = test_data[DF_COLUMN_LANG]
    for index, row in test_data.iterrows():
        scores = {
            lang: sum([get_token_score(token, ngrams, lang, vocab_size)
                       for token in tokenize(row[DF_COLUMN_TWEET], n)])
            for lang in LANGUAGES
        }
        guess = max(scores, key=scores.get)
        results.loc[index, DF_COLUMN_GUESS] = guess
        results.loc[index, DF_COLUMN_SCORE] = scores[guess]
    return results


def get_token_score(token, ngrams, lang, vocab_size) -> float:
    """
    Returns the score of a token given a model.

    Handles situations where then token isn't fully present in the model.
    :param token: The token to score
    :param ngrams: The model
    :param lang: The language
    :param vocab_size: The size of the vocabulary
    :return: The token's score
    """
    score = log10(1 / vocab_size)
    if token[:-1] in ngrams.ngrams[lang].index:
        column = token[-1:] if token[-1:] in ngrams.ngrams[lang].columns else DF_COLUMN_OOV
        score = log10(ngrams.ngrams[lang].loc[token[:-1], column])
    return score
