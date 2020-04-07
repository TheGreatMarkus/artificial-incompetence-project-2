import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from constants import *


def accuracy(results: pd.DataFrame):
    """
    Calculates the accuracy on results.
    :param results: Dataframe of tested results on model.
    :return: accuracy of the model.
    """
    acc = (results[DF_COLUMN_LABEL] == CORRECT_LABEL).sum() / len(results.index)
    return acc


def precision(results: pd.DataFrame):
    """
    Calculates the precision of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages precision.
    """
    pre = {}
    correct = results.loc[results[DF_COLUMN_LABEL] == CORRECT_LABEL]
    wrong = results.loc[results[DF_COLUMN_LABEL] == WRONG_LABEL]

    for language in LANGUAGES:
        true_pos = (correct[DF_COLUMN_GUESS] == language).sum()
        false_pos = (wrong[DF_COLUMN_GUESS] == language).sum()
        if true_pos == 0:
            pre[language] = 0
        else:
            pre[language] = (true_pos / (true_pos + false_pos))
    return pre


def recall(results: pd.DataFrame):
    """
    Calculates the recall of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages recall.
    """
    correct = results.loc[(results[DF_COLUMN_LABEL] == CORRECT_LABEL)]
    rec = {}
    for language in LANGUAGES:
        true_pos = (correct[DF_COLUMN_ACTUAL] == language).sum()
        if (results[DF_COLUMN_ACTUAL] == language).sum() == 0:
            rec[language] = 0
        else:
            rec[language] = (true_pos / (results[DF_COLUMN_ACTUAL] == language).sum())
    return rec


def f1_measure(results: pd.DataFrame):
    """
    Calculates the F1 Measure of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages F1 Measure.
    """
    f1 = {}
    pre = precision(results)
    rec = recall(results)
    for language in LANGUAGES:
        if rec[language] == 0:
            f1[language] = 0
        else:
            f1[language] = (2 * ((pre[language] * rec[language]) / (pre[language] + rec[language])))
    return f1


def macro_f1(results: pd.DataFrame):
    """
    Calculates the Macro F1 Measure on the results.
    :param results: Dataframe of tested results on model.
    :return: the Macro F1 Measure.
    """

    f1 = f1_measure(results)
    macroF1 = 0

    for language in LANGUAGES:
        macroF1 += f1[language]

    return macroF1 / len(LANGUAGES)


def weighted_f1(results: pd.DataFrame):
    """
    Calculates the Weighted Average F1 Measure on the results.
    :param results: Dataframe of tested results on model.
    :return: the Weighted Average F1 Measure.
    """
    f1 = f1_measure(results)
    weightedF1 = 0

    for language in LANGUAGES:
        weightedF1 += (f1[language] * (results[DF_COLUMN_ACTUAL] == language).sum())

    return weightedF1 / len(results.index)


def generate_confusion_matrix(results: pd.DataFrame, illustrate: bool = False):
    """
    Generate confusion matrix output to console.
    Optionally, generate a graphical plot.
    :param results: DataFrame of results.
    :param illustrate: generate plot if true else skip.
    :return:
    """
    confmat = confusion_matrix(y_true=results[DF_COLUMN_ACTUAL], y_pred=results[DF_COLUMN_GUESS],
                               labels=LANGUAGES)
    if illustrate:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        ax.set_xticklabels([''] + LANGUAGES)
        ax.set_yticklabels([''] + LANGUAGES)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], horizontalalignment='center', verticalalignment='center')
        plt.xlabel('Predicted labels')
        plt.ylabel('Actual labels')
        plt.show()
    print('Confusion matrix:\n {}'.format(confmat))


def format_results(results: pd.DataFrame):
    pre = precision(results)
    rec = recall(results)
    f1 = f1_measure(results)

    a, p, r, f, m, w = '', '', '', '', '', ''
    for language in LANGUAGES:
        p += str(EVALUATION_FORMAT.format(pre[language])) + OUTPUT_FILE_SPACE_COUNT * ' '
        r += str(EVALUATION_FORMAT.format(rec[language])) + OUTPUT_FILE_SPACE_COUNT * ' '
        f += str(EVALUATION_FORMAT.format(f1[language])) + OUTPUT_FILE_SPACE_COUNT * ' '
    a += str(EVALUATION_FORMAT.format(accuracy(results))) + OUTPUT_FILE_SPACE_COUNT * ' ' + END_OF_LINE
    m += str(EVALUATION_FORMAT.format(macro_f1(results))) + OUTPUT_FILE_SPACE_COUNT * ' '
    w += str(EVALUATION_FORMAT.format(weighted_f1(results))) + OUTPUT_FILE_SPACE_COUNT * ' ' + END_OF_LINE
    p += END_OF_LINE
    r += END_OF_LINE
    f += END_OF_LINE
    print("Accuracy: " + a)
    print("Precision: " + p)
    print("Recall: " + r)
    print("F1 Measure: " + f)
    print("Macro/Weighted F1 Measure: " + m + w)
    generate_confusion_matrix(results=results, illustrate=False)
    final_format = a + p + r + f + m + w

    return final_format


def evaluate_results(results: pd.DataFrame, v: int, n: int, d: float):
    """
    Records the accuracy, precision, recall, F1 Measure, Weighted/Average F1 Measure of results to .txt file.
    :param results: Dataframe of tested results on model.
    :param v: Vocabulary option
    :param n: ngram of size n
    :param d: Delta smoothing value.
    :return: void.
    """

    if not os.path.exists(EVALUATION_FOLDER):
        os.makedirs(EVALUATION_FOLDER)
    with open(EVALUATION_RESULTS.format(v, n, d), "w") as file:
        file.write(format_results(results))
        file.close()
