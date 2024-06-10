import pandas as pd
import os

dir = r'Z:\Data\YouTube\Anotacje'

df = pd.DataFrame()
for file in os.listdir(dir):
    df_cur = pd.read_csv(os.path.join(dir, file))
    df = pd.concat([df, df_cur], axis = 0)

df = df.reset_index(drop = True)
# drop nans from textDisplay
df = df.dropna(subset = ['textDisplay'])

# save
df.to_csv(r'Z:\Data\YouTube\Anotacje\all.csv', index = False)
df = df[['textOriginal', 'authorDisplayName', 'id', 'videoId']]
df.columns = ['text', 'author', 'id', 'videoid']
df.to_csv(r'data\youtube.csv', index = False)

# load
df = pd.read_csv(r'data\youtube.csv')


