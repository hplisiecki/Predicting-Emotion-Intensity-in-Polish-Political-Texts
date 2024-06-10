import pandas as pd
import numpy as np
from tqdm import tqdm
import pingouin as pg

# Load the data
df = pd.read_csv('data/for_training/ICC_data.csv')
emotions = ['Happiness', 'Sadness', 'Anger', 'Disgust', 'Pride', 'Fear', 'Valence', 'Arousal']


# Function to prepare data for ICC calculation
columns = ['id', 'annotator']

def prepare_data(df, emotion):

    # Add an auxiliary column that counts occurrences of each ['annotator', 'id'] pair
    df['aux_index'] = df.groupby(['annotator', 'id']).cumcount()

    # Set a multi-index including 'annotator', 'id', and the auxiliary column
    temp_df = df.set_index(['annotator', 'id', 'aux_index'])

    # Pivot the DataFrame while keeping all values
    pivoted_df = temp_df.unstack(level=['id', 'aux_index'])[emotion]

    df['aux_index'] = df.groupby(['annotator', 'id']).cumcount()
    temp_df = df.set_index(['annotator', 'id', 'aux_index'])

    pivoted_df.reset_index(inplace=True, drop=True)

    column_list = []
    for col in pivoted_df.columns:
        col_cleaned = pivoted_df[col].dropna()
        # reset index
        col_cleaned = col_cleaned.reset_index(drop = True)
        column_list.append(col_cleaned)

    temp_df = pd.DataFrame(column_list).T

    # reshape into long format
    temp_df = temp_df.melt(var_name='id', value_name=emotion)
    # random
    temp_df['annotator'] = [i%5 for i in range(len(temp_df))]
    # random
    return temp_df


# Bootstrap function to calculate ICC1 and ICC1k and their confidence intervals
def bootstrap_icc(df, emotion, icc_type='ICC1', n_bootstraps=1000, confidence_level=0.95):
    icc_values = []

    # Bootstrap loop
    for _ in tqdm(range(n_bootstraps)):
        # Resample with replacement
        resampled_df = df.sample(n=len(df), replace=True)
        # Prepare data
        temp_df = prepare_data(resampled_df, emotion)
        # Calculate ICC
        icc = pg.intraclass_corr(data=temp_df, targets='id', raters='annotator', ratings=emotion)
        icc_value = icc.set_index('Type').iloc[0]['ICC']  # Assuming the first row corresponds to the desired ICC type
        icc_values.append(icc_value)

    # Calculate confidence intervals
    lower_bound = np.percentile(icc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(icc_values, (1 + confidence_level) / 2 * 100)
    return np.mean(icc_values), lower_bound, upper_bound


# Calculate bootstrap ICC1/ICC1k and confidence intervals for each emotion
for emotion in emotions:
    # ICC1
    mean_icc1, lower_ci1, upper_ci1 = bootstrap_icc(df, emotion, icc_type='ICC1')
    print(f"{emotion} ICC1: Mean = {mean_icc1:.3f}, CI = [{lower_ci1:.3f}, {upper_ci1:.3f}]")

    # ICC1k
    mean_icc1k, lower_ci1k, upper_ci1k = bootstrap_icc(df, emotion, icc_type='ICC1k')
    print(f"{emotion} ICC1k: Mean = {mean_icc1k:.3f}, CI = [{lower_ci1k:.3f}, {upper_ci1k:.3f}]")
