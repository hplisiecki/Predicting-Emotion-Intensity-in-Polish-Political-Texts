import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from tqdm import tqdm

df = pd.read_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered.csv')
# dropna from stemmed
df = df.dropna(subset = ['stemmed_sentence'])



# load
df = pd.read_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered.csv')

df = df[df['language'] == 'pl']

# load stopwords from txt
with open(r'D:\PycharmProjects\annotations\data\stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

texts = [simple_preprocess(text) for text in df['stemmed_sentence']]
# stopwords
texts = [[word for word in text if word not in stopwords] for text in texts]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
# save
model.wv.save(r'D:\PycharmProjects\annotations\models\vectors_words.kv')
model.save(r'D:\PycharmProjects\annotations\models\doc2vec.model')
model.dv.save(r'D:\PycharmProjects\annotations\models\vectors_docs.kv')