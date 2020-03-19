# -----------------------------------------------------------
# main.py 2020-03-19
#
# File serialization 
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

import os.path
import pandas as pd
from constants import *
from ngram import Ngram

def ifExists(vocab: int, ngram: int):
    if(os.path.exists('trainingResults/'+LANG_EN+(str(vocab))+'_'+(str(ngram))+'.pkl')
    and os.path.exists('trainingResults/'+LANG_CA+(str(vocab))+'_'+(str(ngram))+'.pkl')
    and os.path.exists('trainingResults/'+LANG_ES+(str(vocab))+'_'+(str(ngram))+'.pkl')
    and os.path.exists('trainingResults/'+LANG_EU+(str(vocab))+'_'+(str(ngram))+'.pkl')
    and os.path.exists('trainingResults/'+LANG_GL+(str(vocab))+'_'+(str(ngram))+'.pkl')
    and os.path.exists('trainingResults/'+LANG_PT+(str(vocab))+'_'+(str(ngram))+'.pkl')):
        return True
    else:
        return False
    

def loadNgrams(vocab: int, ngram: int):
    ngrams =Ngram(ngram)
    ngrams.ngrams[LANG_EU] = pd.read_pickle('trainingResults/'+LANG_EU+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_CA] = pd.read_pickle('trainingResults/'+LANG_CA+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_GL] = pd.read_pickle('trainingResults/'+LANG_GL+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_ES] = pd.read_pickle('trainingResults/'+LANG_ES+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_EN] = pd.read_pickle('trainingResults/'+LANG_EN+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_PT] = pd.read_pickle('trainingResults/'+LANG_PT+(str(vocab))+'_'+(str(ngram))+'.pkl')
    return ngrams


def saveNgrams(ngrams: Ngram, vocab: int, ngram: int):
    ngrams.ngrams[LANG_EU].to_pickle('trainingResults/'+LANG_EU+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_CA].to_pickle('trainingResults/'+LANG_CA+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_GL].to_pickle('trainingResults/'+LANG_GL+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_ES].to_pickle('trainingResults/'+LANG_ES+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_EN].to_pickle('trainingResults/'+LANG_EN+(str(vocab))+'_'+(str(ngram))+'.pkl')
    ngrams.ngrams[LANG_PT].to_pickle('trainingResults/'+LANG_PT+(str(vocab))+'_'+(str(ngram))+'.pkl')