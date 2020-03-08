# -----------------------------------------------------------
# constants.py 2020-03-08
#
# Define constants for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------
# Languages
LANG_EU = "eu"
LANG_CA = "ca"
LANG_GL = "gl"
LANG_ES = "es"
LANG_EN = "en"
LANG_PT = "pt"

IGNORE_DELIM = '*'

# Dataframe columns
DF_COLUMN_ID = 'ID'
DF_COLUMN_NAME = 'Name'
DF_COLUMN_LANG = 'Languages'
DF_COLUMN_TWEET = 'Tweet'


# Vocabulary
VOCABULARY_0 = 'V0'
VOCABULARY_1 = 'V1'
VOCABULARY_2 = 'V2'

#Ngram
UNIGRAM = 'unigram'
BIGRAM = 'bigram'
TRIGRAM = 'trigram'

class Tweet:
    tweet_id: int
    user: str
    language: str
    tweet: str

    def __init__(self, tweet_id: int, user: str, language: str, tweet: str):
        self.tweet_id = tweet_id
        self.user = user
        self.language = language
        self.tweet = tweet
