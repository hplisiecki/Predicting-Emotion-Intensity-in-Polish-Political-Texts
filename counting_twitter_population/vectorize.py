# load
import os
import pandas as pd

dir = r'Z:\Discrete emotions'

tweets = pd.read_csv(os.path.join(dir, 'all_tweets.csv'))

unique_ids = tweets['author_id'].unique()
# save to csv
df = pd.DataFrame({'author_id': unique_ids})
df.to_csv('Piotr/unique_ids.csv')