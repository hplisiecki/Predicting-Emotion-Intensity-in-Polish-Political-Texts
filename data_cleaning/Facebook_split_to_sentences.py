import pandas as pd
from tqdm import tqdm
# set to not display warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read in data
df_j = pd.read_csv('data/facebook_posts_j.csv')
df_n = pd.read_csv('data/facebook_posts_n.csv')
df_p = pd.read_csv('data/facebook_posts_p.csv')

# concat
df = pd.concat([df_j, df_n, df_p], axis = 0)
df = df.dropna(subset = ['text'])

# replace links (words that start with http or www) with '__link__'
df['text'] = df['text'].apply(lambda x: ' '.join([word if 'http' not in word and 'www' not in word else '__link__' for word in x.split(' ')]))
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
# dropna from text


from nltk.tokenize import sent_tokenize


df['sentences'] = [sentence_tokenizer.tokenize(text) for text in df['text']]

# flatten
df = df.explode('sentences')
df = df.dropna(subset = ['sentences'])

# drop nans
# replace \n

# strip
df['sentences'] = df['sentences'].apply(lambda x: x.strip())
# reset index
# drop ''
df = df[df['sentences'] != '']
df = df.reset_index(drop = True)

# save
df.to_csv(r'data\facebook_posts_sentences.csv', index = False)