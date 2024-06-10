import os
import pandas as pd
from bs4 import BeautifulSoup
from pyMorfologik import Morfologik
from pyMorfologik.parsing import ListParser
import pickle
from tqdm import tqdm
import os
import nltk
import requests
# tokenizer polish
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os
import nltk
import pandas as pd
import numpy as np
import time

with open('data/lemmatizer_dictionary.pickle', 'rb') as handle:
    lema_dict = pickle.load(handle)

dir = 'https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt'
# download the file
r = requests.get(dir)

stopwords = r.text.splitlines()

# with open('data/stopwords.txt', 'wb') as handle:
#     handle.write(r.content)
# # read the file
# with open('data/stopwords.txt', 'r') as f:
#     stopwords = f.read().splitlines()

# stopwords.extend(['http', '@'])

parser = ListParser()
stemmer = Morfologik()
import string
def stem(sentence):
    # remove interpunction
    if sentence[:2] == 'RT':
        sentence = sentence[2:]
    sentence = "".join([ch for ch in sentence if ch not in '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'])
    words = str(sentence).split(' ')
    words = [word for word in words if word not in stopwords and '@' not in word and 'http' not in word]
    # if len(words) == 0:
    #     print('empty')
    tweet = ' '.join(words)
    morf = stemmer.stem([tweet.lower()], parser)
    string = ''
    for i in morf:
        if i[0] in lema_dict.keys():
            string += lema_dict[i[0]] + ' '
        else:
            try:
                string += list(i[1].keys())[0] + ' '
            except:
                string += i[0] + ' '
    string = string[:-1]

    return string

if __name__ == '__main__':

    df = pd.read_csv(r'D:\PycharmProjects\annotations\data\youtube_manual_discrete_sentences.csv')

    RT = True
    if RT:
        # texts = df.text.to_list()
        # new_texts = []
        # rt = []
        # for text in texts:
        #     if text[:2] == 'RT':
        #         new_texts.append(' '.join(text.split(': ')[1:]))
        #         rt.append(True)
        #     else:
        #         new_texts.append(text)
        #         rt.append(False)
        texts_dates = [str(text) for text in df['sentences']]
        new_texts = texts_dates

    else:
        texts_dates = [str(text) for text in df['sentences'] if text[:2] != 'RT']
        new_texts = [text for text in texts_dates]
    print(len(texts_dates))
    print('loaded Tweets')

    texts_set = list(set(new_texts))
    print(len(texts_set))

    pool = multiprocessing.Pool(16)
    L = [pool.map(stem, texts_set)]
    print('stemmed')

    dictionary = dict(zip(texts_set, [line for line in L[0]]))
    # dates = [line[2] for line in texts_dates]
    print('dictionary created')

    # ids = [line[0] for line in texts_dates]
    texts = [dictionary[line] for line in texts_dates]
    pool.close()
    del pool
    # save
    # pydict = {'id': ids, 'text': texts, 'date': dates, 'rt': rt}
    df['stemmed_sentence'] = texts
    # overwrite
    df.to_csv(r'D:\PycharmProjects\annotations\data\youtube_manual_discrete_sentences_stemmed.csv', index=False)
    # load
