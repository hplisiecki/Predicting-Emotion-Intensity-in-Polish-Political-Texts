import pandas as pd
import numpy as np
from tqdm import tqdm
df = pd.read_csv('data/for_training/wyniki_dla_piotra.csv')
emotions = ['Happiness', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Valence', 'Arousal']

text_agreement = {'text': [], 'Happiness': [], 'Sadness': [], 'Anger': [], 'Disgust': [], 'Fear': [], 'Valence': [], 'Arousal': []}
for text in tqdm(df['text'].unique()):
    temp_df = df[df['text'] == text]
    text_agreement['text'].append(text)

    for emotion in emotions:
        text_agreement[emotion].append(np.std(temp_df[emotion]))



text_agreement = pd.DataFrame(text_agreement)

# save
text_agreement.to_csv('data/for_training/text_agreement.csv', index = False)

for emotion in emotions:
    print("Average std for emotion: ", emotion, " is: ", text_agreement[emotion].mean())


import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
df = pd.read_csv('data/for_training/wyniki_dla_piotra.csv')
emotions = ['Happiness', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Valence', 'Arousal']

print('Calculating baseline...')
baseline = {'text': [], 'Happiness': [], 'Sadness': [], 'Anger': [], 'Disgust': [], 'Fear': [], 'Valence': [], 'Arousal': []}
for text in tqdm(df['text'].unique()):
    temp_df = df[df['text'] == text]
    baseline['text'].append(text)
    # ranomly delete n annotators
    # get all combinations of n annotators
    for emotion in emotions:
        baseline[emotion].append(np.mean(temp_df[emotion]))

baseline = pd.DataFrame(baseline)

for n in range(2,5):
    text_distance = {'text': [], 'Happiness': [], 'Sadness': [], 'Anger': [], 'Disgust': [], 'Fear': [], 'Valence': [], 'Arousal': []}
    print("Calculating scenario N: ", n)
    for text in df['text'].unique():
        temp_df = df[df['text'] == text]
        text_distance['text'].append(text)
        # get all combinations of n annotators
        combinations = list(itertools.combinations(temp_df['annotator'].unique(), n))
        for emotion in emotions:
            distance = []
            for combination in combinations:
                temp_temp_df = temp_df[~temp_df['annotator'].isin(combination)]
                distance.append(abs(np.mean(temp_temp_df[emotion]) - baseline[baseline['text'] == text][emotion].values[0]))
            text_distance[emotion].append(np.mean(distance))

    text_distance = pd.DataFrame(text_distance)

    for emotion in emotions:
        print("Average distance for emotion: ", emotion, " is: ", text_distance[emotion].mean())

