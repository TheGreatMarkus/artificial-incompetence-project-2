from constants import *
import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, precision_score


def accuracy(results: pd.DataFrame):
    """
    Calculates the accuracy on results.
    :param results: Dataframe of tested results on model.
    :return: accuracy of the model.
    """
    accuracy = ((results[DF_COLUMN_LABEL] == CORRECT_LABEL).sum()/(results[DF_COLUMN_LABEL] != '').sum())
    return EVALUATION_FORMAT.format(accuracy) + END_OF_LINE

def precision(results: pd.DataFrame):
    """
    Calculates the precision of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages precision.
    """
    precision = ''
    correct = results.loc[(results[DF_COLUMN_LABEL] == CORRECT_LABEL)]
    wrong = results.loc[(results[DF_COLUMN_LABEL] == WRONG_LABEL)]

    for language in LANGUAGES:
        truePos = (correct[DF_COLUMN_ACTUAL] == language).sum()
        falsePos = (wrong[DF_COLUMN_GUESS] == language).sum()
        precision += str(EVALUATION_FORMAT.format(truePos/(truePos + falsePos))) + OUTPUT_FILE_SPACE_COUNT * ' '
    return precision + END_OF_LINE


def recall(results: pd.DataFrame):
    """
    Calculates the recall of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages recall.
    """
    correct = results.loc[(results[DF_COLUMN_LABEL] == CORRECT_LABEL)]
    recall = ''
    for language in LANGUAGES:
        truePos = (correct[DF_COLUMN_ACTUAL] == language).sum()
        recall += str(EVALUATION_FORMAT.format((truePos/(results[DF_COLUMN_ACTUAL] == language).sum()))) + OUTPUT_FILE_SPACE_COUNT * ' '
    return recall + END_OF_LINE

def f1_measure(results: pd.DataFrame):
    """
    Calculates the F1 Measure of each language on the results.
    :param results: Dataframe of tested results on model.
    :return: string of each languages F1 Measure.
    """
    f1 = ''
    correct = results.loc[(results[DF_COLUMN_LABEL] == CORRECT_LABEL)]
    wrong = results.loc[(results[DF_COLUMN_LABEL] == WRONG_LABEL)]
    for language in LANGUAGES:
        truePos = (correct[DF_COLUMN_ACTUAL] == language).sum()
        falsePos = (wrong[DF_COLUMN_GUESS] == language).sum()
        precision = (truePos/(truePos + falsePos))
        recall = truePos/(results[DF_COLUMN_ACTUAL] == language).sum()
        if(recall == 0):
            f1 += str(EVALUATION_FORMAT.format(0)) + OUTPUT_FILE_SPACE_COUNT * ' '
        else:
            f1 += str(EVALUATION_FORMAT.format((2 * ((precision * recall)/(precision + recall))))) + OUTPUT_FILE_SPACE_COUNT * ' '
    return f1 + END_OF_LINE

def macro_and_weighted_f1(results: pd.DataFrame):
    """
    Calculates the Macro F1 Measure and the Weighted Average F1 Measure on the results.
    :param results: Dataframe of tested results on model.
    :return: string of the Macro F1 Measure and the Weighter Average F1 Measure.
    """
    macroF1 = 0
    weightedF1 = 0
    correct = results.loc[(results[DF_COLUMN_LABEL] == CORRECT_LABEL)]
    wrong = results.loc[(results[DF_COLUMN_LABEL] == WRONG_LABEL)]
    for language in LANGUAGES:
        truePos = (correct[DF_COLUMN_ACTUAL] == language).sum()
        falsePos = (wrong[DF_COLUMN_GUESS] == language).sum()
        precision = (truePos / (truePos + falsePos))
        recall = truePos / (results[DF_COLUMN_ACTUAL] == language).sum()
        if (recall == 0):
            macroF1 += 0
            weightedF1 += 0
        else:
            macroF1 += (2 * ((precision * recall) / (precision + recall)))
            weightedF1 += ((2 * ((precision * recall) / (precision + recall)))* (results[DF_COLUMN_ACTUAL] == language).sum())

    return str(EVALUATION_FORMAT.format((macroF1)/len(LANGUAGES))) + OUTPUT_FILE_SPACE_COUNT * ' ' + str(EVALUATION_FORMAT.format((weightedF1/len(results.index))))


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
    if not os.path.exists(EVALUATION_FOLDER):
        os.makedirs(EVALUATION_FOLDER)
    with open(EVALUATION_RESULTS.format(v,n,d), "w") as file:
        file.write(acc)
        file.write(pre)
        file.write(rec)
        file.write(f1)
        file.write(mac_weigh_f1)
        file.close()
    print("Accuracy: " + acc)
    print("Precision: " + pre)
    print("Recall: " + rec)
    print("F1 Measure: " + f1)
    print("Macro/Weighted F1 Measure: " + mac_weigh_f1)