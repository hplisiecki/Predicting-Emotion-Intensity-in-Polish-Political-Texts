from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import pandas as pd
import random

# load
dv = KeyedVectors.load(r'D:\PycharmProjects\annotations\models\vectors_docs.kv', mmap='r')
vectors = [dv[i] for i in dv.index_to_key]

kmeans = KMeans(n_clusters=10)
kmeans.fit(vectors)
labels = kmeans.labels_

df = pd.read_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered.csv')

df = df[df['language'] == 'pl']
# reset index
df = df.reset_index(drop=True)

chosen_vectors = []
for i in range(10):
    cluster_indexes = [j for j, x in enumerate(labels) if x == i]
    # sample one
    next = False

    # sample 10 random sentences
    sample = random.sample(cluster_indexes, 10)


    chosen_vectors.extend(sample)




# pick the chosen texts
picked_texts = df[df.index.isin(chosen_vectors)]
# save
picked_texts.to_csv(r'D:\PycharmProjects\annotations\data\picked_texts.csv', index=False)from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import pandas as pd
import random

# load
dv = KeyedVectors.load(r'D:\PycharmProjects\annotations\models\vectors_docs.kv', mmap='r')
vectors = [dv[i] for i in dv.index_to_key]

kmeans = KMeans(n_clusters=10)
kmeans.fit(vectors)
labels = kmeans.labels_

df = pd.read_csv(r'D:\PycharmProjects\annotations\data\facebook_posts_sentences_filtered.csv')

df = df[df['language'] == 'pl']
# reset index
df = df.reset_index(drop=True)

chosen_vectors = []
for i in range(10):
    cluster_indexes = [j for j, x in enumerate(labels) if x == i]
    # sample one
    next = False

    # sample 10 random sentences
    sample = random.sample(cluster_indexes, 10)


    chosen_vectors.extend(sample)




# pick the chosen texts
picked_texts = df[df.index.isin(chosen_vectors)]
# save
picked_texts.to_csv(r'D:\PycharmProjects\annotations\data\picked_texts.csv', index=False)