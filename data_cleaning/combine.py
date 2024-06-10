import pandas as pd
import os
from langdetect import detect
from tqdm import tqdm

youtube = pd.read_csv(r'data\youtube.csv')
youtube['user_type'] = 'not_applicable'
youtube['source'] = 'youtube'

languages = []
for i in tqdm(youtube['text']):
    try:
        languages.append(detect(i))
    except:
        languages.append('error')
        print(i)
youtube['language'] = languages

# save
youtube.to_csv(r'data\youtube.csv', index = False)
dir = r'Z:\Discrete emotions'
tweets = pd.read_csv(os.path.join(dir, 'all_tweets.csv'))
tweets['source'] = 'twitter'
languages = []
for i in tqdm(tweets['text']):
    try:
        languages.append(detect(i))
    except:
        languages.append('error')
        print(i)

tweets['language'] = languages
# save
tweets.to_csv(os.path.join(dir, 'all_tweets.csv'), index = False)

# load
tweets = pd.read_csv(os.path.join(dir, 'all_tweets.csv'))

facebook = pd.read_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered.csv')
facebook['source'] = 'facebook'
facebook['full_post'] = facebook['text']
facebook['text'] = facebook['sentence']
del facebook['sentence']



