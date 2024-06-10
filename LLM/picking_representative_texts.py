
######################################################################
########################## ONESHOT ###################################
######################################################################
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

def select_representative_row(df, embeddings_col, emotion_col):
    """
    Selects a row that is in the middle of the mass of the vectors in the embeddings column
    and in the middle of the annotation scale for the specified emotion column.

    Parameters:
    - df: pandas.DataFrame containing the dataset.
    - embeddings_col: str, the name of the column containing the embeddings vectors.
    - emotion_col: str, the name of the emotion column to consider.

    Returns:
    - selected_row: pandas.Series representing the selected row.
    """

    # Step 1: Compute the center of mass for embeddings
    embeddings_array = np.stack(df[embeddings_col].values)  # Convert embeddings column to a numpy array
    center_of_mass = np.mean(embeddings_array, axis=0)  # Compute the mean embedding vector

    # Step 2: Calculate the distance of each embedding from the center of mass
    distances = np.linalg.norm(embeddings_array - center_of_mass, axis=1)  # Euclidean distance
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())  # Normalize distances

    # Step 3: Compute centrality for the specified emotion
    emotional_centrality = abs(df[emotion_col] - 0.5)  # Deviation from the middle of the scale
    normalized_emotional_centrality = (emotional_centrality - emotional_centrality.min()) / (
                emotional_centrality.max() - emotional_centrality.min())  # Normalize

    # Step 4: Combine semantic and emotional scores
    combined_score = 0.5 * normalized_distances + 0.5 * normalized_emotional_centrality  # Assuming equal weights

    # Step 5: Select the row with the lowest combined score
    selected_row_index = combined_score.idxmin()
    selected_row = df.iloc[selected_row_index]

    return selected_row

df = pd.read_csv('data/embeddings/train_set.csv')
df['embeddings'] = df['embeddings'].apply(json.loads)



# Usage example:
# Assuming 'df' is your DataFrame, 'embeddings' is the column with vector embeddings,
# and 'norm_Happiness_M' is the emotion column you're focusing on.

emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M']

collate = []
for emotion in tqdm(emotion_columns):
    selected_row = select_representative_row(df, 'embeddings', emotion)
    collate.append(selected_row)

# create dataframe
selected_rows = pd.DataFrame(collate)
# add emotions
selected_rows['emotion'] = emotion_columns
# save
selected_rows.to_csv('data/embeddings/one_shot.csv', index=False)

######################################################################
########################## TWOSHOT ###################################
######################################################################
import numpy as np
import pandas as pd
import json
from tqdm import tqdm


def select_emotion_exemplars(df, embeddings_col, emotion_col):
    """
    Selects positive and negative exemplars based on vector centrality and emotional value.
    Positive exemplar is closer to the highest emotional rating, and negative exemplar is closer to the lowest.

    Parameters:
    - df: pandas.DataFrame containing the dataset.
    - embeddings_col: str, the name of the column containing the embeddings vectors.
    - emotion_col: str, the name of the emotion column to consider.

    Returns:
    - positive_exemplar, negative_exemplar: pandas.Series representing the selected rows.
    """

    # Compute the center of mass for embeddings
    embeddings_array = np.stack(df[embeddings_col].values)
    center_of_mass = np.mean(embeddings_array, axis=0)

    # Calculate the distance of each embedding from the center of mass and normalize
    distances = np.linalg.norm(embeddings_array - center_of_mass, axis=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Emotional value adjustment: Positive scores for high ratings, negative for low ratings
    # For positive exemplar: Use raw emotional ratings as weights
    positive_weights = df[emotion_col]

    # For negative exemplar: Invert the emotional ratings to prioritize lower ratings
    negative_weights = 1 - df[emotion_col]  # Inverting the scale

    # Combine vector centrality and emotional weights for selection
    positive_score = 0.5 * normalized_distances - 0.5 * positive_weights  # Subtracting to prioritize higher emotional ratings
    negative_score = 0.5 * normalized_distances - 0.5 * negative_weights  # Subtracting inverted weights to prioritize lower ratings

    # Select the rows with the lowest combined scores
    positive_exemplar_index = positive_score.idxmin()
    negative_exemplar_index = negative_score.idxmin()

    positive_exemplar = df.iloc[positive_exemplar_index]
    negative_exemplar = df.iloc[negative_exemplar_index]

    return positive_exemplar, negative_exemplar


df = pd.read_csv('data/embeddings/train_set.csv')
df['embeddings'] = df['embeddings'].apply(json.loads)



# Usage example:
# Assuming 'df' is your DataFrame, 'embeddings' is the column with vector embeddings,
# and 'norm_Happiness_M' is the emotion column you're focusing on.

emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M']


collate = []
for emotion in tqdm(emotion_columns):
    selected_rows = select_emotion_exemplars(df, 'embeddings', emotion)
    collate.append(selected_rows)

# flatten
collate = [item for sublist in collate for item in sublist]
# create dataframe hstack
selected_rows = pd.DataFrame(collate)

# add emotions
selected_rows['emotion'] = [emotion for emotion in emotion_columns for _ in (0, 1)]

# save
selected_rows.to_csv('data/embeddings/two_shot.csv', index=False)


######################################################################
########################## FOURSHOT LISTWISE #########################
######################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def select_fourshot_emotion_exemplars(df, embeddings_col, emotion_col):
    """
    Selects four exemplars based on quartiles of emotional value and vector centrality.
    Texts are chosen to represent the first (Q1), second (Q2), third (Q3), and fourth (Q4) quartiles of emotion.

    Parameters:
    - df: pandas.DataFrame containing the dataset.
    - embeddings_col: str, the name of the column containing the embeddings vectors.
    - emotion_col: str, the name of the emotion column to consider.

    Returns:
    - A tuple of pandas.Series representing the selected rows for Q1, Q2, Q3, and Q4 emotions.
    """

    # Compute the center of mass for embeddings
    embeddings_array = np.stack(df[embeddings_col].values)
    center_of_mass = np.mean(embeddings_array, axis=0)

    # Calculate the distance of each embedding from the center of mass and normalize
    distances = np.linalg.norm(embeddings_array - center_of_mass, axis=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Determine quartiles based on emotional value
    Q1, Q2, Q3 = df[emotion_col].quantile([0.25, 0.5, 0.75]).values

    # Define quartile datasets
    quartile_dfs = [
        df[df[emotion_col] <= Q1],
        df[(df[emotion_col] > Q1) & (df[emotion_col] <= Q2)],
        df[(df[emotion_col] > Q2) & (df[emotion_col] <= Q3)],
        df[df[emotion_col] > Q3]
    ]

    exemplars = []
    for quartile_df in quartile_dfs:
        print(len(quartile_df))
        # Calculate normalized distances for each quartile
        quartile_distances = normalized_distances[quartile_df.index]

        # Select the row with the minimum distance in each quartile
        exemplar_index = quartile_distances.argmin()
        exemplars.append(quartile_df.iloc[exemplar_index])

    return tuple(exemplars)

df = pd.read_csv('data/embeddings/train_set.csv')
df['embeddings'] = df['embeddings'].apply(json.loads)


# Usage example:
# Assuming 'df' is your DataFrame, 'embeddings' is the column with vector embeddings,
# and 'norm_Happiness_M' is the emotion column you're focusing on.

emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M']


collate = []
for emotion in tqdm(emotion_columns):
    selected_rows = select_fourshot_emotion_exemplars(df, 'embeddings', emotion)
    collate.append(selected_rows)

# flatten
collate = [item for sublist in collate for item in sublist]
# create dataframe hstack
selected_rows = pd.DataFrame(collate)

selected_rows['emotion'] = [emotion for emotion in emotion_columns for _ in range(4)]

# save
selected_rows.to_csv('data/embeddings/four_shot.csv', index=False)

# load
selected_rows = pd.read_csv('data/embeddings/four_shot.csv')


######################################################################
########################## FOURSHOT four points ######################
######################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

def select_emotion_exemplars_four_points(df, embeddings_col, emotion_col):
    """
    Selects exemplars based on vector centrality and proximity to specified emotional scale points (0.2, 0.4, 0.6, 0.8).

    Parameters:
    - df: pandas.DataFrame containing the dataset.
    - embeddings_col: str, the name of the column containing the embeddings vectors.
    - emotion_col: str, the name of the emotion column to consider.

    Returns:
    - A dictionary with keys '0.2', '0.4', '0.6', '0.8' and values being pandas.Series representing the selected rows for each point.
    """

    # Compute the center of mass for embeddings
    embeddings_array = np.stack(df[embeddings_col].values)
    center_of_mass = np.mean(embeddings_array, axis=0)

    # Calculate the distance of each embedding from the center of mass and normalize
    distances = np.linalg.norm(embeddings_array - center_of_mass, axis=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Initialize a dictionary to hold the exemplars
    exemplars = []

    # Define target emotional points
    emotional_points = [0.2, 0.4, 0.6, 0.8]

    for point in emotional_points:
        # Calculate the absolute difference between the emotional rating and the target point
        emotional_differences = np.abs(df[emotion_col] - point)

        # Combine vector centrality and emotional differences for selection
        combined_score = 0.5 * normalized_distances + 0.5 * emotional_differences  # Adding to prioritize proximity to the target point

        # Select the row with the lowest combined score for this point
        exemplar_index = combined_score.idxmin()
        exemplars.append(df.iloc[exemplar_index])


    return exemplars

df = pd.read_csv('data/embeddings/train_set.csv')
df['embeddings'] = df['embeddings'].apply(json.loads)

emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M']


collate = []
for emotion in tqdm(emotion_columns):
    selected_rows = select_emotion_exemplars_four_points(df, 'embeddings', emotion)
    collate.append(selected_rows)

# flatten
collate = [item for sublist in collate for item in sublist]
# create dataframe hstack
selected_rows = pd.DataFrame(collate)

selected_rows['emotion'] = [emotion for emotion in emotion_columns for _ in range(4)]

# save
selected_rows.to_csv('data/embeddings/four_shot_four_points.csv', index=False)


##################################################################################
########################## FIVESHOT FIVE POINTS ##################################
##################################################################################

import numpy as np
import pandas as pd
import json
from tqdm import tqdm


def select_emotion_exemplars_five_points(df, embeddings_col, emotion_col):
    """
    Selects exemplars based on vector centrality and proximity to specified emotional scale points (1/6, 2/6, 3/6, 4/6, 5/6).

    Parameters:
    - df: pandas.DataFrame containing the dataset.
    - embeddings_col: str, the name of the column containing the embeddings vectors.
    - emotion_col: str, the name of the emotion column to consider.

    Returns:
    - A dictionary with keys '1/6', '2/6', '3/6', '4/6', '5/6' and values being pandas.Series representing the selected rows for each point.
    """

    # Compute the center of mass for embeddings
    embeddings_array = np.stack(df[embeddings_col].values)
    center_of_mass = np.mean(embeddings_array, axis=0)

    # Calculate the distance of each embedding from the center of mass and normalize
    distances = np.linalg.norm(embeddings_array - center_of_mass, axis=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Initialize a dictionary to hold the exemplars
    exemplars = []

    # Define target emotional points based on the 1/6 scale increments
    emotional_points = [i/6 for i in range(1, 6)]

    for point in emotional_points:
        # Calculate the absolute difference between the emotional rating and the target point
        emotional_differences = np.abs(df[emotion_col] - point)

        # Combine vector centrality and emotional differences for selection
        combined_score = 0.5 * normalized_distances + 0.5 * emotional_differences  # Adding to prioritize proximity to the target point

        # Select the row with the lowest combined score for this point
        exemplar_index = combined_score.idxmin()
        exemplars.append(df.iloc[exemplar_index])

    return exemplars

df = pd.read_csv('data/embeddings/train_set.csv')
df['embeddings'] = df['embeddings'].apply(json.loads)

emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M']


collate = []
for emotion in tqdm(emotion_columns):
    selected_rows = select_emotion_exemplars_five_points(df, 'embeddings', emotion)
    collate.append(selected_rows)

# flatten
collate = [item for sublist in collate for item in sublist]
# create dataframe hstack
selected_rows = pd.DataFrame(collate)

selected_rows['emotion'] = [emotion for emotion in emotion_columns for _ in range(5)]

# save
selected_rows.to_csv('data/embeddings/five_shot_five_points.csv', index=False)


######################################################################
########################## FIVESHOT ##################################
######################################################################





import numpy as np
import pandas as pd
from tqdm import tqdm
import json


def select_fiveshot_emotion_exemplars_listwise(df, embeddings_col, emotion_col):
    """
    Selects five exemplars based on listwise (equal-frequency) splits of emotional value and vector centrality.
    Ensures each split contains an equal number of observations.

    Parameters:
    - df: pandas.DataFrame containing the dataset.
    - embeddings_col: str, the name of the column containing the embeddings vectors.
    - emotion_col: str, the name of the emotion column to consider.

    Returns:
    - A list of pandas.Series representing the selected rows for each of the five splits.
    """

    # Compute the center of mass for embeddings
    embeddings_array = np.stack(df[embeddings_col].values)
    center_of_mass = np.mean(embeddings_array, axis=0)

    # Calculate the distance of each embedding from the center of mass and normalize
    distances = np.linalg.norm(embeddings_array - center_of_mass, axis=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Sort dataframe by emotional value
    sorted_df = df.sort_values(by=emotion_col).reset_index(drop=True)

    # Split dataframe into five equal parts
    split_dfs = np.array_split(sorted_df, 5)

    exemplars = []
    for split_df in split_dfs:
        # Get the indices of the split in the original dataframe
        original_indices = split_df.index

        # Calculate normalized distances for the split
        split_distances = normalized_distances[original_indices]

        # Select the row with the minimum distance in the split
        exemplar_index = original_indices[split_distances.argmin()]
        exemplars.append(df.iloc[exemplar_index])

    return exemplars

    return tuple(exemplars)
df = pd.read_csv('data/embeddings/train_set.csv')
df['embeddings'] = df['embeddings'].apply(json.loads)


# Usage example:
# Assuming 'df' is your DataFrame, 'embeddings' is the column with vector embeddings,
# and 'norm_Happiness_M' is the emotion column you're focusing on.

emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M']


collate = []
for emotion in tqdm(emotion_columns):
    break
    selected_rows = select_fiveshot_emotion_exemplars_listwise(df, 'embeddings', emotion)
    collate.append(selected_rows)

# flatten
collate = [item for sublist in collate for item in sublist]
# create dataframe hstack
selected_rows = pd.DataFrame(collate)

selected_rows['emotion'] = [emotion for emotion in emotion_columns for _ in range(5)]

# save
selected_rows.to_csv('data/embeddings/five_shot.csv', index=False)