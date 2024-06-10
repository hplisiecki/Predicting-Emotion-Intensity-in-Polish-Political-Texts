import pandas as pd
import os
from tqdm import tqdm
import numpy as np
df_list = []

for i in tqdm(range(100)):
    annotations_df = pd.read_csv(f'data/annotations/Z{i+1}.csv', sep = ';')
    texts_df = pd.read_csv(f'data/picked/picked_all_{i+1}.csv')
    columns = annotations_df.columns
    # get index of specific column
    index = columns.get_loc('Całkowity czas')
    columns = columns[:index]
    annotators = annotations_df.iloc[:, 0].unique()
    columns = columns[1:]
    questions_len = len(columns) / len(texts_df)
    column_names = ['Happiness', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Pride', 'Valence', 'Arousal', 'Irony']
    results = []
    temp_results = []
    counter = 1
    for col in columns:
        temp_results.append(annotations_df[col].tolist())
        if counter % questions_len == 0:
            results.append(temp_results)
            counter = 0
            temp_results = []
        counter += 1

    for i, annotator in enumerate(annotators):
        for j, col_name in enumerate(column_names):
            texts_df[f'{col_name}_{annotator}'] = [res[j][i] for res in results]

    for col_name in column_names:
        if col_name == 'Irony':
            # for each count the number of 'Y' and give a score
            texts_df[f'{col_name}_M'] = texts_df[[f'{col_name}_{annotator}' for annotator in annotators]].apply(lambda x: x.value_counts().get('Y', 0), axis = 1)
            continue
        texts_df[f'{col_name}_M'] = texts_df[[f'{col_name}_{annotator}' for annotator in annotators]].mean(axis = 1)

    # save
    texts_df.to_csv(f'data/after_merging/merged_{i+1}.csv', index = False)
    retain_columns = ['text', 'source', 'part', 'Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M', 'Irony_M']
    texts_df = texts_df[retain_columns]
    df_list.append(texts_df)


df = pd.concat(df_list)
df.to_csv('data/for_training/merged.csv', index = False)



from sklearn.model_selection import train_test_split
import pandas as pd
# load
df = pd.read_csv('data/for_training/merged.csv')

# get all duplicate rows:
duplicate_rows = df[df.duplicated(subset=['text'], keep=False)]
for text in duplicate_rows['text'].unique():
    temp = duplicate_rows[duplicate_rows['text'] == text]
    columns = temp.columns
    columns = columns[3:]
    # averages of all emotions
    new_row = {'text': text, 'source': temp['source'].iloc[0], 'part': str(temp['part'].iloc[0]) + '_' +  str(temp['part'].iloc[1]), 'Happiness_M': temp['Happiness_M'].mean(), 'Sadness_M': temp['Sadness_M'].mean(), 'Anger_M': temp['Anger_M'].mean(), 'Disgust_M': temp['Disgust_M'].mean(), 'Fear_M': temp['Fear_M'].mean(), 'Pride_M': temp['Pride_M'].mean(), 'Valence_M': temp['Valence_M'].mean(), 'Arousal_M': temp['Arousal_M'].mean(), 'Irony_M': temp['Irony_M'].mean()}
    df = df[df['text'] != text]
    # concat
    df = pd.concat([df, pd.DataFrame(new_row, index=[0])])


emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M', 'Irony_M']

# histograms for each of the emotions
import matplotlib.pyplot as plt
for col in emotion_columns:
    plt.hist(df[col], bins = 20)
    plt.title(col)
    plt.show()
    # save
    plt.savefig(f'plots/{col}.png')
    plt.close()

for emotion in emotion_columns:
    df[f'z_score_{emotion}'] = np.abs((df[emotion] - df[emotion].mean()) / df[emotion].std(ddof=0))

df['total_z_score'] = df[[f'z_score_{emotion}' for emotion in emotion_columns]].sum(axis = 1)

# sort
df = df.sort_values(by=['total_z_score'], ascending=False)
df['emotion_sum'] = df[[f'{emotion}' for emotion in emotion_columns]].sum(axis = 1)
# hist
import matplotlib.pyplot as plt

plt.hist(df['emotion_sum'], bins=20)
plt.title('emotion_sum')
plt.show()
# save
plt.savefig(f'plots/emotion_sum.png')
plt.close()




# weighted sample
test_set = df.sample(frac=0.1, weights=df['total_z_score'], random_state=1)
test_set['emotion_sum'] = test_set[[f'{emotion}' for emotion in emotion_columns]].sum(axis = 1)
plt.hist(test_set['emotion_sum'], bins=20)
plt.title('emotion_sum')
plt.show()
# save
plt.savefig(f'plots/test/emotion_sum.png')
plt.close()

# save test set
test_set.to_csv('data/for_training/test_set.csv', index = False)

# load
df = pd.read_csv('data/for_training/merged.csv')
train_set = df[~df['text'].isin(test_set['text'])]
train, val = train_test_split(train_set, test_size=0.111, random_state=1)
train.to_csv('data/for_training/train_set.csv', index = False)
val.to_csv('data/for_training/val_set.csv', index = False)



for col in emotion_columns:
    plt.hist(test_set[col], bins = 20)
    plt.title(col)
    plt.show()
    # save
    plt.savefig(f'plots/test/{col}.png')
    plt.close()

