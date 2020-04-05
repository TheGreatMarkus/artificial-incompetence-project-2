from functools import reduce
from typing import Dict, List

import pandas as pd
from nltk.lm import Lidstone
from nltk.lm.preprocessing import padded_everygram_pipeline

from constants import *
from custom_tokenize import tokenize
from evaluate import format_results
from test_results import prepare_result_df, finalize_result_df
from utils import validate_params
from vocabulary import transform_to_vocab

LIDSTONE_LEFT_PAD_SYMBOL = '<s>'
LIDSTONE_RIGHT_PAD_SYMBOL = '</s>'
LIDSTONE_TOKENIZE_SPAN = 1


def argmax(models: Dict[str, Lidstone], tweet: List[str]):
    """
    Get argmax score and a corresponding language
    :param models: Dictionary of trained models by language
    :param tweet: Tweet instance
    :return: Tuple of best score and corresponding language
    """
    best_lang = ''
    best_score = float('-inf')
    for lang, model in models.items():
        score = reduce(lambda log_sum, ngram: log_sum + model.logscore(ngram[-1:][0], ngram[:-1]), tweet, 0)
        best_lang = lang if score > best_score else best_lang
        best_score = max(best_score, score)
    return best_score, best_lang


def modify_padding(ngram_char: str):
    """
    NLTK uses special sequence of chars for left and right padding.
    Primary target of this NLTK library is to work with words,
    hence working with characters requires padding adjustment
    :param ngram_char: ngram characters
    :return: modified character if NLTK padding, else no change
    """
    if ngram_char == LEFT_PAD_SYMBOL:
        return LIDSTONE_LEFT_PAD_SYMBOL
    elif ngram_char == RIGHT_PAD_SYMBOL:
        return LIDSTONE_RIGHT_PAD_SYMBOL
    return ngram_char


def lidstone(v: int, n: int, gamma: float, train_file: str, test_file: str):
    """
    Provides Lidstone-smoothed scores.
    In addition to initialization arguments from BaseNgramModel
    also requires a number by which to increase the counts, gamma.
    :param v: Vocabulary choice
    :param n: ngram choice
    :param gamma: Smoothing choice
    :param train_file: Path to training data
    :param test_file: Path to testing data
    :return:
    """
    validate_params(v, n, gamma, train_file, test_file)

    # Process train data
    train_data = pd.read_csv(train_file,
                             delimiter='\t',
                             names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])
    train_data.drop(labels=[DF_COLUMN_ID, DF_COLUMN_NAME], inplace=True, axis=1)
    transform_to_vocab(train_data, v)
    train_data[DF_COLUMN_TWEET] = train_data[DF_COLUMN_TWEET].map(lambda tweet: tokenize(tweet, LIDSTONE_TOKENIZE_SPAN))

    # Train model
    models_by_lang = {}
    for language, tweets in train_data.groupby(DF_COLUMN_LANG)[DF_COLUMN_TWEET]:
        tweet_list = tweets.tolist()
        train_ngrams, padded_vocab = padded_everygram_pipeline(n, tweet_list)
        model = Lidstone(gamma=gamma, order=n)
        model.fit(train_ngrams, padded_vocab)
        models_by_lang[language] = model

    # Process test data
    test_data = pd.read_csv(test_file, delimiter='\t',
                            names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])
    transform_to_vocab(test_data, v)
    test_data[DF_COLUMN_TWEET] = test_data[DF_COLUMN_TWEET].map(lambda tweet: tokenize(tweet=tweet, span=n,
                                                                                       extended_func=True))
    test_data[DF_COLUMN_TWEET] = test_data[DF_COLUMN_TWEET].map(
        lambda tweet_ngrams: [[modify_padding(ngram_char) for ngram_char in list(ngram)] for ngram in tweet_ngrams])

    # Calculate scores
    test_data[DF_COLUMN_TWEET] = test_data[DF_COLUMN_TWEET].map(lambda tweet_ngrams: argmax(models_by_lang,
                                                                                            tweet_ngrams))
    score_lang_df = pd.DataFrame(test_data[DF_COLUMN_TWEET].tolist(), columns=[DF_COLUMN_SCORE, DF_COLUMN_GUESS])

    # Finalize results
    results = prepare_result_df(test_data, score_lang_df)
    results = finalize_result_df(results)

    # Evaluation stats
    print("Evaluating Lidstone with parameters: [vocabulary = {}, ngram size = {}, delta = {}]".format(v, n, gamma))
    format_results(results)
