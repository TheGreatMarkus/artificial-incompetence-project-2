from math import log10

import numpy as np
import pandas as pd

from constants import *
from custom_tokenize import tokenize
from ngram import Ngram


def prior_probability(test_data: pd.DataFrame, lang: str):
    count_langs = test_data[DF_COLUMN_LANG].value_counts()
    if lang not in count_langs.index or count_langs[lang] == 0:
        return 0
    return log10(count_langs[lang] / len(test_data.index))


def get_test_results(test_data: pd.DataFrame, ngrams: Ngram, vocab_size: int, n: int) -> pd.DataFrame:
    """
    Generate test result DataFrame from test data and model
    :param test_data: The test data to evaluate
    :param ngrams: The models
    :param vocab_size: The size of the vocabulary
    :param n: The size of the ngrams
    :return: Result DataFrame
    """
    results = prepare_result_df(test_data)
    for index, row in test_data.iterrows():
        scores = {
            lang: prior_probability(test_data, lang) + sum(
                [get_token_score(token, ngrams, lang, vocab_size)
                 for token in tokenize(row[DF_COLUMN_TWEET], n)])
            for lang in LANGUAGES
        }
        guess = max(scores, key=scores.get)
        results.loc[index, DF_COLUMN_GUESS] = guess
        results.loc[index, DF_COLUMN_SCORE] = scores[guess]
    results = finalize_result_df(results)
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
        score = ngrams.ngrams[lang].loc[token[:-1], column]
        if score == 0:
            return float('-inf')
        score = log10(score)
    return score


def prepare_result_df(test_data: pd.DataFrame, score_language_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Prepare layout of the result DataFrame.
    Score and Guess can be fetched optionally from already generated `score_language_df`.
    :param score_language_df: DataFrame that contains scores and corresponding languages
    :param test_data: test data DataFrame.
    :return: partially filled result DataFrame.
    """
    results = pd.DataFrame(columns=[DF_COLUMN_ID, DF_COLUMN_GUESS, DF_COLUMN_SCORE, DF_COLUMN_ACTUAL, DF_COLUMN_LABEL])
    results[DF_COLUMN_ID] = test_data[DF_COLUMN_ID]
    results[DF_COLUMN_ACTUAL] = test_data[DF_COLUMN_LANG]
    if score_language_df is not None:
        results[DF_COLUMN_GUESS] = score_language_df[DF_COLUMN_GUESS]
        results[DF_COLUMN_SCORE] = score_language_df[DF_COLUMN_SCORE]
    return results


def finalize_result_df(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare result DataFrame for submission.
    - Transform `Guess` column to scientific notation.
    - Generate values for the Label column.
    :param result_df: result DataFrame
    :return: updated DataFrame
    """
    result_df[DF_COLUMN_SCORE] = result_df[DF_COLUMN_SCORE].map(
        lambda score_val: format(score_val, SCIENTIFIC_NOTATION_FORMAT))
    result_df[DF_COLUMN_LABEL] = np.where(result_df[DF_COLUMN_GUESS] == result_df[DF_COLUMN_ACTUAL],
                                          CORRECT_LABEL, WRONG_LABEL)
    return result_df
