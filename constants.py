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

# Ngram DataFrame columns, besides necessary features
DF_COLUMN_SUM = 'Sum'

# Ngram
UNIGRAM = 1
BIGRAM = 2
TRIGRAM = 3

# Vocabulary
VOCABULARY_0 = 0
VOCABULARY_1 = 1
VOCABULARY_2 = 2

# Serialization
TRAINING_RESULT_FOLDER = 'trainingResults'
TRAINING_FILE_TEMPLATE = TRAINING_RESULT_FOLDER + '/{}{}_{}.pkl'

