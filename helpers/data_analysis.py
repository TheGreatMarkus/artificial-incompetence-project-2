import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.extend([os.getcwd()])  # So that code works in terminal

from constants import DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET

# Absolute path
data_file_name = "/abs/path/to/file.txt"
chart_title = "Chart title"

file_data = pd.read_csv(data_file_name, delimiter='\t',
                        names=[DF_COLUMN_ID, DF_COLUMN_NAME, DF_COLUMN_LANG, DF_COLUMN_TWEET])

num_tweets = len(file_data.index)
lang_count = file_data[DF_COLUMN_LANG].value_counts()

print("Data set has {} tweets.".format(num_tweets))
for lang, num_lang in lang_count.iteritems():
    percent = num_lang / num_tweets * 100
    avg_tweet_len = file_data.loc[file_data[DF_COLUMN_LANG] == lang][DF_COLUMN_TWEET].transform(len).mean()
    print(
        "Language \"{}\" has {} tweets, which is {:.2f}% of all tweets"
        ", and has a mean length of {:.0f} characters.".format(lang, num_lang, percent, avg_tweet_len))

x = lang_count.index.values
values = lang_count.values
x_pos = [i for i in range(len(x))]
plt.bar(x, values)
plt.xlabel("Languages")
plt.ylabel("# Tweets")
plt.title(chart_title)

plt.show()
