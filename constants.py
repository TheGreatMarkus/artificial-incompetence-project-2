# -----------------------------------------------------------
# constants.py 2020-03-08
#
# Define constants for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

# Delimiter for out of vocabulary characters
OUT_OF_VOCABULARY_DELIM = '*'

# Languages
LANG_EU = "eu"
LANG_CA = "ca"
LANG_GL = "gl"
LANG_ES = "es"
LANG_EN = "en"
LANG_PT = "pt"

LANGUAGES = [LANG_EU, LANG_CA, LANG_GL, LANG_ES, LANG_EN, LANG_PT]

# Input DataFrame columns
DF_COLUMN_ID = 'ID'
DF_COLUMN_NAME = 'Name'
DF_COLUMN_LANG = 'Language'
DF_COLUMN_TWEET = 'Tweet'

# Ngram DataFrame columns
DF_COLUMN_OOV = 'OOV'

# Score DataFrame columns
DF_COLUMN_GUESS = 'guess'
DF_COLUMN_SCORE = 'score'
DF_COLUMN_ACTUAL = 'actual'
DF_COLUMN_LABEL = 'label'

# Ngram
UNIGRAM = 1
BIGRAM = 2
TRIGRAM = 3

# Vocabulary
VOCABULARY_0 = 0
VOCABULARY_1 = 1
VOCABULARY_2 = 2

VOCABULARY_0_SIZE = 26
VOCABULARY_1_SIZE = 52
VOCABULARY_2_SIZE = 116766

# Serialization
TRAINING_RESULT_FOLDER = 'trainingResults'
TRAINING_FILE_TEMPLATE = TRAINING_RESULT_FOLDER + '/{}_{}-{}-{}.pkl'
EVALUATION_FOLDER = 'evaluation'
EVALUATION_RESULTS = EVALUATION_FOLDER + '/eval_{}_{}_{}.txt'

# Labels
CORRECT_LABEL = 'correct'
WRONG_LABEL = 'wrong'

# Trace file
TRACE_FILE_DIR = 'trace_files'
TRACE_FILE_TEMPLATE = TRACE_FILE_DIR + '/trace_{}_{}_{}.txt'
SCIENTIFIC_NOTATION_FORMAT = '.2E'
OUTPUT_FILE_SPACE_COUNT = 2

# Hyperparam validation
VALID_VOCABULARIES = [VOCABULARY_0, VOCABULARY_1, VOCABULARY_2]
VALID_NGRAMS = [UNIGRAM, BIGRAM, TRIGRAM]

VOCABULARY_VALUE_ERROR_MESSAGE = 'Vocabulary input does not match any valid vocabulary'
NGRAM_VALUE_ERROR_MESSAGE = 'Ngram input does not match any valid ngram'
DELTA_VALUE_ERROR_MESSAGE = 'The value of delta is out of (0 ... 1] range'
MISSING_TRAIN_FILE_ERROR_MESSAGE = 'Train file does not exist'
MISSING_TEST_FILE_ERROR_MESSAGE = 'Test file does not exist'

#Evaluation
EVALUATION_FORMAT = "{:0<.4f}"
END_OF_LINE = '\r'