from constants import *
import pandas as pd
import os

def accuracy(results: pd.DataFrame):
    """
    Calculates the accuracy on results.
    :param results: Dataframe of tested results on model.
    :return: accuracy of the model.
    """
    accuracy = ((results.label == CORRECT_LABEL).sum()/(results.label != '').sum())
    return EVALUATION_FORMAT.format(accuracy) + '\r'

def precision(results: pd.DataFrame):
    """
    Calculates the precision of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages precision.
    """
    precision = ''
    correct = results.loc[(results.label == CORRECT_LABEL)]
    wrong = results.loc[(results.label == WRONG_LABEL)]

    for language in LANGUAGES:
        truePos = ( correct.actual == language).sum()
        falsePos = (wrong.actual == language).sum()
        precision += str(EVALUATION_FORMAT.format(truePos/(truePos + falsePos))) + '  '
    return precision + '\r'


def recall(results: pd.DataFrame):
    """
    Calculates the recall of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages recall.
    """
    correct = results.loc[(results.label == CORRECT_LABEL)]
    recall = ''
    for language in LANGUAGES:
        truePos = (correct.actual == language).sum()
        recall += str(EVALUATION_FORMAT.format((truePos/(results.actual == language).sum()))) + '  '
    return recall + '\r'

def f1_Measure(results: pd.DataFrame):
    """
    Calculates the F1 Measure of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages F1 Measure.
    """
    f1 = ''
    correct = results.loc[(results.label == CORRECT_LABEL)]
    wrong = results.loc[(results.label == WRONG_LABEL)]
    for language in LANGUAGES:
        truePos = (correct.actual == language).sum()
        falsePos = (wrong.actual == language).sum()
        precision = (truePos/(truePos + falsePos))
        recall = truePos/(results.actual == language).sum()
        if(recall == 0):
            f1 += '0.0000  '
        else:
            f1 += str(EVALUATION_FORMAT.format((2 * ((precision * recall)/(precision + recall))))) + '  '
    return f1 + '\r'

def macro_And_Weighted_F1(results: pd.DataFrame):
    """
    Calculates the Macro F1 Measure and the Weighted Average F1 Measure on the results.
    :param results: Dataframe of tested results on model.
    :return: string of the Macro F1 Measure and the Weighter Average F1 Measure.
    """
    macroF1 = 0
    weightedF1 = 0
    correct = results.loc[(results.label == CORRECT_LABEL)]
    wrong = results.loc[(results.label == WRONG_LABEL)]
    for language in LANGUAGES:
        truePos = (correct.actual == language).sum()
        falsePos = (wrong.actual == language).sum()
        precision = (truePos / (truePos + falsePos))
        recall = truePos / (results.actual == language).sum()
        if (recall == 0):
            macroF1 += 0
            weightedF1 += 0
        else:
            macroF1 += (2 * ((precision * recall) / (precision + recall)))
            weightedF1 += ((2 * ((precision * recall) / (precision + recall)))* (results.actual == language).sum())

    return str(EVALUATION_FORMAT.format((macroF1)/len(LANGUAGES))) + '  ' + str(EVALUATION_FORMAT.format((weightedF1/len(results.index))))


def evaluate_Results(results: pd.DataFrame, v:int,n:int, d:float):
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
    file = open(EVALUATION_RESULTS.format(v,n,d), "w")
    file.write(accuracy(results))
    file.write(precision(results))
    file.write(recall(results))
    file.write(f1_Measure(results))
    file.write(macro_And_Weighted_F1(results))
    file.close()