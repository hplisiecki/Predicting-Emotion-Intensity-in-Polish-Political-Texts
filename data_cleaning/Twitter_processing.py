import os
import pandas as pd
import zipfile
from tqdm import tqdm
dir = r'Z:\Discrete emotions'
# unzip files in dir
for file in os.listdir(dir):
    if '.zip' in file:
        with zipfile.ZipFile(os.path.join(dir, file), 'r') as zip_ref:
            zip_ref.extractall(dir)

types = ['journalists', 'ngos', 'politicians']
for type in types:
    print(type)
    type_collate = pd.DataFrame()
    type_modded = 'tweets_' + type
    for directory in os.listdir(os.path.join(dir, type_modded)):
        print(directory)
        for file in tqdm(os.listdir(os.path.join(dir, type_modded, directory))):
            if 'users' in file or 'query' in file:
                continue
            df = pd.read_json(os.path.join(dir, type_modded, directory, file))
            df = df[['text', 'public_metrics', 'id', 'author_id', 'created_at', 'lang']]
            type_collate = pd.concat([type_collate, df], axis = 0)

    # save
    type_collate.to_csv(os.path.join(dir, type, f'{type}_tweets.csv'), index = False)

all_df = pd.DataFrame()
for type in types:
    curr_df = pd.read_csv(os.path.join(dir, type, f'{type}_tweets.csv'))
    curr_df['user_type'] = type
    all_df = pd.concat([all_df, curr_df], axis = 0)

# save
all_df.to_csv(os.path.join(dir, 'all_tweets.csv'), index = False)

# load
import os
dir = r'Z:\Discrete emotions'

tweets = pd.read_csv(os.path.join(dir, 'all_tweets.csv'))



# load
youtube = pd.read_csv(r'data\youtube.csv')