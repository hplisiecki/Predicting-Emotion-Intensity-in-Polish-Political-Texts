import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/facebook_posts_sentences.csv')

# replace multiple spaces with a single space
df['sentences'] = df['sentences'].apply(lambda x: ' '.join(x.split()))

months = 'January February March April May June July August September October November December'.split(' ')
df = df[~df['sentences'].str.contains('|'.join(months))]

df = df[df['sentences'].str.len() > 5]
# drop all sentences that contain a month

df = df.reset_index(drop = True)

for idx in tqdm(df.index):
    if df['sentences'][idx][-1] == '…' or df['sentences'][idx][-4:] == 'More':
        if df['id'][idx + 1] != df['id'][idx]:
            df = df.drop(idx)

df = df[df['sentences'].str.split(' ').apply(len) > 1]

df = df.drop_duplicates(subset = ['sentences'])

df = df.reset_index(drop = True)

# save
df.to_csv('data/facebook_posts_sentences_filtered.csv', index = False)

# load
from langdetect import detect, DetectorFactory
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('data/facebook_posts_sentences_filtered.csv')

df = df[df['sentences'].str.len() < 500]

DetectorFactory.seed = 0

languages = []
for i in tqdm(df['sentences']):
    try:
        languages.append(detect(i))
    except:
        languages.append('error')
        print(i)

df['language'] = languages

df = df[df['language'] == 'pl']

df.to_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered.csv', index=False)
