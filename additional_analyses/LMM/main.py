import pandas as pd

df = pd.read_csv('data/for_training/ICC_data.csv')

df_encoded = pd.get_dummies(df, columns=['annotator'], drop_first=True)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_encoded[['Happiness']] = scaler.fit_transform(df_encoded[['Happiness']])


import torch

# Fixed effects design matrix
X_fixed = torch.tensor(df_encoded[['intercept', 'fixed_effect_1', 'fixed_effect_2']].values, dtype=torch.float)

# Subject indices for random effects
# Ensure subjects are coded as consecutive integers starting from 0
subjects = pd.Categorical(df_encoded['subject_id'])
subject_indices = torch.tensor(subjects.codes, dtype=torch.long)  # Convert subject IDs to indices

# Design matrix for variable(s) with random slopes
X_random_slope = torch.tensor(df_encoded[['random_slope_variable']].values, dtype=torch.float)
