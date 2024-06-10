import pandas as pd
from tqdm import tqdm
df = pd.read_csv('data/for_training/ICC_data.csv')

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations

emotions = ['Happiness', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Valence', 'Arousal']


# Find unique raters
raters = df['annotator'].unique()

# Generate all unique pairs of raters
rater_pairs = list(combinations(raters, 2))

# Initialize a list to store kappa scores
kappa_scores = []

# Calculate Cohen's Kappa for each pair
for rater1, rater2 in tqdm(rater_pairs):
    # Find common items rated by both raters
    ids1 = df[df['annotator'] == rater1]['id'].unique()
    ids2 = df[df['annotator'] == rater2]['id'].unique()

    common_items = list(set(ids1).intersection(ids2))

    # Filter ratings for these common items
    ratings1 = [    df[ (df['annotator'] == rater1)  & (df['id'] == id_)]['Happiness'].iloc[0]    for id_ in common_items    ]
    ratings2 = [    df[ (df['annotator'] == rater2)  & (df['id'] == id_)]['Happiness'].iloc[0]    for id_ in common_items    ]

    # Calculate Cohen's Kappa and store the result
    kappa = cohen_kappa_score(ratings1, ratings2)
    kappa_scores.append((rater1, rater2, kappa))

# Convert the results to a DataFrame
kappa_df = pd.DataFrame(kappa_scores, columns=['Rater1', 'Rater2', 'Cohen_Kappa'])
print(kappa_df)