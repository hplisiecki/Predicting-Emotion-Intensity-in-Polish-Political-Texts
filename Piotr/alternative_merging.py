import pandas as pd
import os
from tqdm import tqdm
import numpy as np


all_results = []
annotators_list = []
collate = []
collate_no = []
for i in tqdm(range(100)):
    annotations_df = pd.read_csv(f'data/annotations/Z{i+1}.csv', sep = ';')

    annotators_list.extend(list(annotations_df.iloc[:, 0].values))



    texts_df = pd.read_csv(f'data/picked/picked_all_{i+1}.csv')
    columns = annotations_df.columns
    # get index of specific column
    index = columns.get_loc('Całkowity czas')
    columns = columns[:index]
    annotators = annotations_df.iloc[:, 0]
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


    vertical_results = []
    for i_, annotator in enumerate(annotators):
        temp_text_df = texts_df.copy()
        for j_, col_name in enumerate(column_names):
            # append row
            temp_text_df[col_name] = [res[j_][i_] for res in results]
        temp_text_df['annotator'] = annotator
        vertical_results.append(temp_text_df)

    vertical_results = pd.concat(vertical_results)

    all_results.append(vertical_results)

from collections import Counter
Counter(annotators_list)


all_results = pd.concat(all_results)

all_results.to_csv('data/for_training/wyniki_dla_piotra.csv', index = False)

# load
