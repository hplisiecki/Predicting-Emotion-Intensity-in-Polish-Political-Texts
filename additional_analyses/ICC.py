import pandas as pd
import numpy as np
from tqdm import tqdm
import pingouin as pg
df = pd.read_csv('data/for_training/ICC_data.csv')
emotions = ['Happiness', 'Sadness', 'Anger', 'Disgust', 'Pride', 'Fear', 'Valence', 'Arousal']

# remove annotator - text duplicates



columns = ['id', 'annotator']
for emotion in emotions:
    temp_columns = columns.copy()
    temp_columns.extend([emotion])
    temp_df = df[temp_columns]
    # transpose so that each column is a text
    temp_df = temp_df.pivot(index='annotator', columns='id', values = emotion)
    column_list = []
    for col in temp_df.columns:
        col_cleaned = temp_df[col].dropna()
        # reset index
        col_cleaned = col_cleaned.reset_index(drop = True)
        column_list.append(col_cleaned)
    temp_df = pd.DataFrame(column_list).T

    # reshape into long format
    temp_df = temp_df.melt(var_name='id', value_name=emotion)
    # random
    temp_df['annotator'] = [i%5 for i in range(len(temp_df))]

    # calculate ICC
    icc = pg.intraclass_corr(data=temp_df, targets= 'id', raters='annotator', ratings=emotion)

    output = icc.set_index('Type')
    print(emotion)
    print(output)
    break


import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming 'df' is your DataFrame with columns 'Subject', 'Rater', and 'Rating'

# Define the model formula; here, 'Rating' is the dependent variable, 'Rater' is treated as a fixed effect,
# and random intercepts are modeled for each 'Subject'
model_formula = 'Happiness ~ C(annotator) + (1|id)'

model = smf.mixedlm(model_formula, data=df, groups=df['id'], re_formula="~0+annotator")
result = model.fit()