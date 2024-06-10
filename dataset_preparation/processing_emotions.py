# on emotion basis
import pandas as pd
from gensim.utils import simple_preprocess
from tqdm import tqdm


# load
df = pd.read_csv(r'/data/youtube_manual_discrete.csv')

from nltk.tokenize import sent_tokenize

df['sentences'] = [sent_tokenize(text) if len(text) > 280 else text for text in df['text']]

# flatten
df = df.explode('sentences')
df = df.dropna(subset = ['sentences'])

df['sentences'] = df['sentences'].apply(lambda x: x.strip())
# reset index
# drop ''
df = df[df['sentences'] != '']
df = df.reset_index(drop = True)

# save
df.to_csv(r'D:\PycharmProjects\annotations\data\youtube_manual_discrete_sentences.csv', index=False)

dir = r'Z:\Discrete emotions'
tweets = pd.read_csv(os.path.join(dir, 'all_tweets.csv'))
tweets = tweets[tweets['language'] == 'pl']
ratings = rate_discrete(tweets['stemmed'])
tweets['happiness'] = ratings[0]
tweets['sadness'] = ratings[1]
tweets['anger'] = ratings[2]
tweets['fear'] = ratings[3]
tweets['disgust'] = ratings[4]
tweets.to_csv(os.path.join(dir, 'all_tweets_manual_discrete.csv'), index=False)

# load


# # weighted sample
# happ['weight'] = happ['happiness'] + happ['sadness'] + happ['anger'] + happ['fear'] + happ['disgust']
# picked = happ.sample(n=300, weights='weight')
#
#
# # save to three different excel files
# picked[:100].to_csv(r'D:\PycharmProjects\annotations\data\picked_texts_1.csv', index=False)
# picked[100:200].to_csv(r'D:\PycharmProjects\annotations\data\picked_texts_2.csv', index=False)
# picked[200:].to_csv(r'D:\PycharmProjects\annotations\data\picked_texts_3.csv', index=False)






############## rating continuous emotions ####################
import pandas as pd
words_full = pd.read_excel("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4947584/bin/DataSheet1.XLSX", sheet_name="Arkusz1", index_col = 0)
words = words_full.loc[:,[col for col in words_full.columns if not ("Male" in col or "Female" in col or
                                                          "MIN" in col or "MAX" in col or "_N" in col)]]

words = words.rename(columns={"part of speach":"part of speech"}) # Poprawka miss spellingu
print(words.columns) #Jakie mamy informacje
words.head()
words = words.set_index("polish word")
emotion_norms = words.loc[:,[col for col in words.columns if "M" in col or "part of speech" in col or 'SD' in col]]# Wybieranie samych średnich ocen

emotion_norms = emotion_norms.drop("part of speech", axis=1)
emotion_norms = (emotion_norms-emotion_norms.mean())/emotion_norms.std() #normalize ratings


emotion_norms['valence_norm'] = (emotion_norms['Valence_M'] - min(emotion_norms['Valence_M'])) / (max(emotion_norms['Valence_M']) - min(emotion_norms['Valence_M']))
emotion_norms['arousal_norm'] = (emotion_norms['arousal_M'] - min(emotion_norms['arousal_M'])) / (max(emotion_norms['arousal_M']) - min(emotion_norms['arousal_M']))
emotion_norms['dominance_norm'] = (emotion_norms['dominance_M'] - min(emotion_norms['dominance_M'])) / (max(emotion_norms['dominance_M']) - min(emotion_norms['dominance_M']))

def rate_discrete(texts):
    ratings = [[] for _ in range(3)]

    for text in tqdm(texts):
        # check if text is a nan
        if type(text) != str:
            for i in range(3):
                ratings[i].append(0)
            continue
        text = simple_preprocess(text)
        text = [word for word in text if word in emotion_norms.index.values]
        if len(text) > 0:
            temp_ratings = [[] for i in range(3)]
            for word in text:
                temp_ratings[0].append(emotion_norms.loc[word]['valence_norm'])
                temp_ratings[1].append(emotion_norms.loc[word]['arousal_norm'])
                temp_ratings[2].append(emotion_norms.loc[word]['dominance_norm'])
            for i in range(3):
                ratings[i].append(sum(temp_ratings[i]) / len(temp_ratings[i]))

        else:
            for i in range(3):
                ratings[i].append(0)
    return ratings


df = pd.read_csv(r'/data/facebook_posts_sentences_filtered_manual_discrete.csv')
results = rate_discrete(df['stemmed_sentence'])
df['valence'] = results[0]
df['arousal'] = results[1]
df['dominance'] = results[2]
df.to_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered_manual_discrete_continuous.csv', index=False)

df = pd.read_csv(r'/data/youtube_manual_discrete_sentences_stemmed.csv')
results = rate_discrete(df['stemmed_sentence'])
df['valence'] = results[0]
df['arousal'] = results[1]
df['dominance'] = results[2]
df.to_csv(r'D:\PycharmProjects\annotations\data\youtube_manual_discrete_nodiscrete_sentences_stemmed_continous.csv', index=False)

dir = r'Z:\Discrete emotions'
tweets = pd.read_csv(os.path.join(dir, 'all_tweets_manual_discrete.csv'))
results = rate_discrete(tweets['stemmed'])
tweets['valence'] = results[0]
tweets['arousal'] = results[1]
tweets['dominance'] = results[2]
tweets.to_csv(os.path.join(dir, 'all_tweets_manual_discrete_continuous.csv'), index=False)


################# picking texts ############################


youtube = pd.read_csv(r'../data/youtube_manual_discrete_continuous.csv')
dir = r'Z:\Discrete emotions'
tweets = pd.read_csv(os.path.join(dir, 'all_tweets_manual_discrete_continuous.csv'))
facebook = pd.read_csv(r'/data/facebook_posts_sentences_filtered_manual_discrete_continuous.csv')

emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust']
picked_yt = pd.DataFrame()
picked_tw = pd.DataFrame()
picked_fb = pd.DataFrame()
for emo in emotions:
    temp_yt = youtube.sample(n = 20, weights=emo)
    temp_tw = tweets.sample(n = 20, weights=emo)
    temp_fb = facebook.sample(n = 20, weights=emo)
    picked_yt = pd.concat([picked_yt, temp_yt])
    picked_tw = pd.concat([picked_tw, temp_tw])
    picked_fb = pd.concat([picked_fb, temp_fb])

# sample once again, randomly
temp_yt = youtube.sample(n = 20)
temp_tw = tweets.sample(n = 20)
temp_fb = facebook.sample(n = 20)
picked_yt = pd.concat([picked_yt, temp_yt])
picked_tw = pd.concat([picked_tw, temp_tw])
picked_fb = pd.concat([picked_fb, temp_fb])

picked_yt.to_csv(r'data\picked_yt_discrete.csv', index=False)
picked_tw.to_csv(r'data\picked_tw_discrete.csv', index=False)
picked_fb.to_csv(r'data\picked_fb_discrete.csv', index=False)

youtube['weight'] = youtube['valence'] + youtube['arousal'] + youtube['dominance']
tweets['weight'] = tweets['valence'] + tweets['arousal'] + tweets['dominance']
facebook['weight'] = facebook['valence'] + facebook['arousal'] + facebook['dominance']

picked_yt = youtube.sample(n = 100, weights='weight')
picked_tw = tweets.sample(n = 100, weights='weight')
picked_fb = facebook.sample(n = 100, weights='weight')

picked_yt.to_csv(r'data\picked_yt_continuous.csv', index=False)
picked_tw.to_csv(r'data\picked_tw_continuous.csv', index=False)
picked_fb.to_csv(r'data\picked_fb_continuous.csv', index=False)


import pandas as pd
import os
youtube = pd.read_csv(r'data/youtube_manual_discrete_nodiscrete_sentences_stemmed_continous.csv')
dir = r'Z:\Discrete emotions'
tweets = pd.read_csv(os.path.join(dir, 'all_tweets_manual_discrete_continuous.csv'))
facebook = pd.read_csv(r'data/facebook_posts_sentences_filtered_manual_discrete_continuous.csv')
print('Facebook:', len(facebook))
print('Tweets:', len(tweets))
print('Youtube:', len(youtube))
# add remove duplicates
facebook = facebook[[True if "wesprzyj.bosak2020.pl" not in text else False for text in facebook['sentences']]]
facebook = facebook[[True if "#FaktyPoFaktach" not in text else False for text in facebook['sentences']]]
facebook = facebook[[True if "NA ŻYWO" not in text else False for text in facebook['sentences']]]

# replace all words that start with @ with _user_ DO SPRAWDZENIA
import re
# replace all multiple occurences of _user_ with one
tweets['text'] = [re.sub(r'@\w+', '_user_ ', text) for text in tweets['text']]
tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]

tweets['text'] = [re.sub(r'_user_\s*(_user_\s*)*', '_users_ ', text) for text in tweets['text']]
tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]

tweets = tweets.drop_duplicates(subset=['text'])
tweets['text'] = [re.sub(r'http\S+', '_link_ ', text) for text in tweets['text']]

tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]
tweets['text'] = [re.sub(r'_link_\s*(_link_\s*)*', '_link_ ', text) for text in tweets['text']]

tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]
tweets['text'] = [text.replace('\n', ' ') for text in tweets['text']]

tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]

youtube['sentences'] = [re.sub(r'@\w+', '_user_ ', text) for text in youtube['sentences']]
youtube['sentences'] = [text.replace('  ', ' ') for text in youtube['sentences']]
# replace all multiple occurences of _user_ with one

youtube['sentences'] = [re.sub(r'_user_\s*(_user_\s*)*', '_users_ ', text) for text in youtube['sentences']]
youtube['sentences'] = [text.replace('  ', ' ') for text in youtube['sentences']]

# drop duplicate texts from all
youtube = youtube.drop_duplicates(subset=['sentences'])
facebook = facebook.drop_duplicates(subset=['sentences'])

# replace all links (http://...) with _link_
youtube['sentences'] = [re.sub(r'http\S+', '_link_ ', text) for text in youtube['sentences']]

youtube['sentences'] = [text.replace('  ', ' ') for text in youtube['sentences']]
# replace all multiple occurences of _user_ with one
youtube['sentences'] = [re.sub(r'_link_\s*(_link_\s*)*', '_link_ ', text) for text in youtube['sentences']]
youtube['sentences'] = [text.replace('  ', ' ') for text in youtube['sentences']]

# replace \n with space
youtube['sentences'] = [text.replace('\n', ' ') for text in youtube['sentences']]
facebook['sentences'] = [text.replace('\n', ' ') for text in facebook['sentences']]
youtube['sentences'] = [text.replace('  ', ' ') for text in youtube['sentences']]
facebook['sentences'] = [text.replace('  ', ' ') for text in facebook['sentences']]


facebook['length'] = [len(text) for text in facebook['sentences']]
youtube['length'] = [len(text) for text in youtube['sentences']]
tweets['length'] = [len(text) for text in tweets['text']]

# sort descending
facebook = facebook.sort_values(by=['length'], ascending=False)
youtube = youtube.sort_values(by=['length'], ascending=False)
tweets = tweets.sort_values(by=['length'], ascending=False)

# drop all with length > 280
facebook = facebook[facebook['length'] <= 280]
tweets = tweets[tweets['length'] <= 280]
youtube = youtube[youtube['length'] <= 280]

youtube['weight'] = youtube['valence'] + youtube['arousal'] + youtube['dominance']
tweets['weight'] = tweets['valence'] + tweets['arousal'] + tweets['dominance']
facebook['weight'] = facebook['valence'] + facebook['arousal'] + facebook['dominance']

facebook['text'] = facebook['sentences']
youtube['text'] = youtube['sentences']


# fb 2719
# tw 4884
# yt 397

# 2000

# nonw fb 680
# nonw tw 1221
# nonw yt 99

picked_yt = youtube.sample(n = 397, weights='weight')
picked_tw = tweets.sample(n = 4884, weights='weight')
picked_fb = facebook.sample(n = 2719, weights='weight')

nonw_youtube = youtube[youtube.index.isin(picked_yt.index) == False]
nonw_tweets = tweets[tweets.index.isin(picked_tw.index) == False]
nonw_facebook = facebook[facebook.index.isin(picked_fb.index) == False]

picked_yt_nonw = nonw_youtube.sample(n = 99, weights='weight')
picked_tw_nonw = nonw_tweets.sample(n = 1221, weights='weight')
picked_fb_nonw = nonw_facebook.sample(n = 680, weights='weight')


youtube = youtube.drop(picked_yt.index)
tweets = tweets.drop(picked_tw.index)
facebook = facebook.drop(picked_fb.index)

picked_all = pd.concat([picked_yt, picked_tw, picked_fb, picked_yt_nonw, picked_tw_nonw, picked_fb_nonw])

# randomly split into 20 equal parts
picked_all = picked_all.sample(frac=1).reset_index(drop=True)
picked_all = picked_all[['text', 'source']]
picked_all['part'] = [i % 100 for i in range(len(picked_all))]
for part in range(100):
    picked_all[picked_all['part'] == part].to_csv(r'data\picked_all_{}.csv'.format(part + 1), index=False)


import pandas as pd
df_list = []
for part in range(100):
    df_list.append(pd.read_csv(r'data\picked\picked_all_{}.csv'.format(part + 1)))

df = pd.concat(df_list)

# load all
import pandas as pd
import os
from tqdm import tqdm
collate = []
for part in range(100):
    collate.append(pd.read_csv(r'data\picked\picked_all_{}.csv'.format(part + 1)))

df = pd.concat(collate)

dir = r'Z:\Discrete emotions'
tweets = pd.read_csv(os.path.join(dir, 'all_tweets_manual_discrete_continuous.csv'))

import re
tweets['text'] = [re.sub(r'@\w+', '_user_ ', text) for text in tweets['text']]
tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]

tweets['text'] = [re.sub(r'_user_\s*(_user_\s*)*', '_users_ ', text) for text in tweets['text']]
tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]

tweets = tweets.drop_duplicates(subset=['text'])
tweets['text'] = [re.sub(r'http\S+', '_link_ ', text) for text in tweets['text']]

tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]
tweets['text'] = [re.sub(r'_link_\s*(_link_\s*)*', '_link_ ', text) for text in tweets['text']]

tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]
tweets['text'] = [text.replace('\n', ' ') for text in tweets['text']]

tweets['text'] = [text.replace('  ', ' ') for text in tweets['text']]

# merge to df on texxt
df = pd.merge(df, tweets, on='text', how='left')

# save
df.to_csv(r'data\weird_merge.csv', index=False)

# Match and merge the DataFrames
result = match_dataframes(df, tweets, 'text', 'text')
print(result)
