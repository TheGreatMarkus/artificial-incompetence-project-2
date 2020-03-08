# -----------------------------------------------------------
# constants.py 2020-03-08
#
# Define constants for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

LANG_EU = "eu"
LANG_CA = "ca"
LANG_GL = "gl"
LANG_ES = "es"
LANG_EN = "en"
LANG_PT = "pt"

IGNORE_DELIM = '*'
DF_COLUMN_TWEET = 'Tweet'

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
