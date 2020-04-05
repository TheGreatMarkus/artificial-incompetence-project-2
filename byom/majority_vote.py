from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from typing import Dict
import re
from utils import validate_params, generate_trace_file
from test_results import prepare_result_df, finalize_result_df
from vocabulary import transform_to_vocab
from evaluate import evaluate_results
from constants import *


def custom_transform_to_vocab(df: pd.DataFrame, v: int) -> None:
    """
    Substitute `*` delim with a space.
    :param df: input DataFrame
    :param v: vocabulary value
    :return: void
    """
    transform_to_vocab(df, v)
    df[DF_COLUMN_TWEET] = df[DF_COLUMN_TWEET].map(lambda tweet: re.sub('\\*+', ' ', tweet))


def encode_class_labels(labels: pd.Series) -> (Dict[str, int], Dict[int, str]):
    """
    Language labels needs to be encoded into numeric values.
    :param labels: DataFrame column that contains all labels.
    :return: tuple of mapping and inverse mapping
    """
    lang_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
    inv_lang_mapping = {val: lang for lang, val in lang_mapping.items()}
    return lang_mapping, inv_lang_mapping


def majority_vote_cl(v: int, n: int, delta: float, train_file: str, test_file: str,
                     min_token_freq: int = 1, max_token_freq: float = 1.0):
    """
    Entry point of program.
    :param max_token_freq: ignore terms that have a document frequency strictly higher than the given proportion.
    :param min_token_freq: ignore terms that have a document frequency strictly lower than the given threshold.
    :param v: Vocabulary choice
    :param n: ngram choice
    :param delta: Smoothing choice
    :param train_file: Path to training data
    :param test_file: Path to testing data
    :return: void
    """
    validate_params(v, n, delta, train_file, test_file)

    # Process data
    train_data = pd.read_csv(train_file,
                             delimiter='\t',
                             names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])
    test_data = pd.read_csv(test_file, delimiter='\t',
                            names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])

    lang_mapping, inv_lang_mapping = encode_class_labels(train_data[DF_COLUMN_LANG])
    train_data[DF_COLUMN_LANG] = train_data[DF_COLUMN_LANG].map(lang_mapping)
    custom_transform_to_vocab(train_data, v)
    custom_transform_to_vocab(test_data, v)

    # Prepare features (Ngrams and their weights)
    tfidf = TfidfVectorizer(analyzer='char_wb', lowercase=False, ngram_range=(n, n),
                            min_df=min_token_freq, max_df=max_token_freq)
    features = tfidf.fit_transform(train_data[DF_COLUMN_TWEET]).toarray()
    labels = train_data[DF_COLUMN_LANG]

    # Define Estimators
    svc = LinearSVC()
    svc_calibrated = CalibratedClassifierCV(svc)
    lr = LogisticRegression(multi_class='multinomial', max_iter=500)
    estimators = [('lr', lr), ('svc_calibrated', svc_calibrated)]

    # Train model
    voting_classifier = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_classifier.fit(features, labels)

    # Calculate scores
    features_test = tfidf.transform(test_data[DF_COLUMN_TWEET])
    guess = voting_classifier.predict(features_test)
    scores = voting_classifier.predict_proba(features_test)

    # Finalize results
    results = prepare_result_df(test_data)
    results[DF_COLUMN_SCORE] = scores
    results[DF_COLUMN_GUESS] = guess
    results[DF_COLUMN_GUESS] = results[DF_COLUMN_GUESS].map(inv_lang_mapping)
    results = finalize_result_df(results)
    generate_trace_file(v, n, delta, results)

    # Evaluation stats
    print(
        "\nEvaluating Majority Vote classifier with parameters: [vocabulary = {}, ngram size = {}, delta = {}]".format(
            v, n, delta))
    evaluate_results(results, v, n, delta)
    return results


majority_vote_cl(VOCABULARY_1, BIGRAM, 0.5, '../training-tweets.txt', '../test-tweets.txt')
