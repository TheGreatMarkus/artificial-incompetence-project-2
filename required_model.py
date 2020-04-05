import pandas as pd

from constants import DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET
from evaluate import evaluate_results
from test_results import get_test_results
from utils import validate_params, process_train_data, generate_trace_file
from vocabulary import get_vocab_size, transform_to_vocab


def required_model(v: int, n: int, delta: float, train_file: str, test_file: str):
    """
    Run the required model with the given parameters
    :param v: Vocabulary choice
    :param n: ngram choice
    :param delta: Smoothing choice
    :param train_file: Path to training data
    :param test_file: Path to testing data
    :return: void
    """
    validate_params(v, n, delta, train_file, test_file)
    vocab_size = get_vocab_size(v)
    ngrams = process_train_data(v, n, delta, vocab_size, train_file)

    test_data = pd.read_csv(test_file, delimiter='\t',
                            names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])
    transform_to_vocab(test_data, v)

    print("Running model against provided testing data.")
    results = get_test_results(test_data, ngrams, vocab_size, n)
    generate_trace_file(v, n, delta, results)

    print("Final results generated")
    print(results)

    print("Evaluating classifier with parameters: [vocabulary = {}, ngram size = {}, delta = {}]".format(v, n, delta))
    evaluate_results(results, v, n, delta)
    return results
