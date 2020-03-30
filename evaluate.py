from constants import *
import pandas as pd
import os



def accuracy(results: pd.DataFrame):
    """
    Calculates the accuracy on results.
    :param results: Dataframe of tested results on model.
    :return: accuracy of the model.
    """
    accuracy = ((results[DF_COLUMN_LABEL] == CORRECT_LABEL).sum()/(results[DF_COLUMN_LABEL] != '').sum())
    return accuracy

def precision(results: pd.DataFrame):
    """
    Calculates the precision of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages precision.
    """
    pre = {}
    correct = results.loc[(results[DF_COLUMN_LABEL] == CORRECT_LABEL)]
    wrong = results.loc[(results[DF_COLUMN_LABEL] == WRONG_LABEL)]

    for language in LANGUAGES:
        true_pos = (correct[DF_COLUMN_ACTUAL] == language).sum()
        false_pos = (wrong[DF_COLUMN_GUESS] == language).sum()
        pre.update({language:(true_pos/(true_pos + false_pos))})
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
        rec.update({language: true_pos/(results[DF_COLUMN_ACTUAL] == language).sum()})
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
            f1.update({language: 0})
        else:
            f1.update({language: (2 * ((pre[language] * rec[language])/(pre[language] + rec[language])))})
    return f1

def macro_and_weighted_f1(results: pd.DataFrame):
    """
    Calculates the Macro F1 Measure and the Weighted Average F1 Measure on the results.
    :param results: Dataframe of tested results on model.
    :return: string of the Macro F1 Measure and the Weighter Average F1 Measure.
    """

    f1 = f1_measure(results)
    macroF1 = 0
    weightedF1 = 0

    for language in LANGUAGES:
        macroF1 += f1[language]
        weightedF1 += (f1[language] * (results[DF_COLUMN_ACTUAL] == language).sum())

    return [macroF1 / len(LANGUAGES), (weightedF1 / len(results.index))]


def evaluate_results(results: pd.DataFrame, v:int,n:int, d:float):
    """
    Records the accuracy, precision, recall, F1 Measure, Weighted/Average F1 Measure of results to .txt file.
    :param results: Dataframe of tested results on model.
    :param v: Vocabulary option
    :param n: ngram of size n
    :param d: Delta smoothing value.
    :return: void.
    """
    acc = accuracy(results)
    pre = precision(results)
    rec = recall(results)
    f1 = f1_measure(results)
    mac_weigh_f1 = macro_and_weighted_f1(results)
    a, p, r, f, m = '', '', '', '', ''
    for language in LANGUAGES:
        p += str(EVALUATION_FORMAT.format(pre[language])) + OUTPUT_FILE_SPACE_COUNT * ' '
        r += str(EVALUATION_FORMAT.format(rec[language])) + OUTPUT_FILE_SPACE_COUNT * ' '
        f += str(EVALUATION_FORMAT.format(f1[language])) + OUTPUT_FILE_SPACE_COUNT * ' '
    a += str(EVALUATION_FORMAT.format(acc)) + OUTPUT_FILE_SPACE_COUNT * ' ' + END_OF_LINE
    p += END_OF_LINE
    r += END_OF_LINE
    f += END_OF_LINE
    for col in mac_weigh_f1:
        m += str(EVALUATION_FORMAT.format(col)) + OUTPUT_FILE_SPACE_COUNT * ' '
    if not os.path.exists(EVALUATION_FOLDER):
        os.makedirs(EVALUATION_FOLDER)
    with open(EVALUATION_RESULTS.format(v,n,d), "w") as file:
        file.write(a)
        file.write(p)
        file.write(r)
        file.write(f)
        file.write(m)
        file.close()
    print("Accuracy: " + a)
    print("Precision: " + p)
    print("Recall: " + r)
    print("F1 Measure: " + f)
    print("Macro/Weighted F1 Measure: " + m)