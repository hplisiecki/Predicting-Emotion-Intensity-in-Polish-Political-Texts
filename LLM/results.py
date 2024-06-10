

# valence and arousal



import pandas as pd
import numpy as np

def is_numeric_or_numeric_string(x):
    try:
        float(x)  # Attempt to convert to float
        return True  # Conversion successful
    except ValueError:
        return False  # Conversion failed, value is not numeric

def check_correlation(test_set, emotion, emocja, verbose = True):
    if verbose:
        print("#########################################")
    rejected = []
    if verbose:
        print(emocja)
    test_set[emocja] = test_set[emocja].apply(lambda x: str(x).replace('"', ''))
    # retain only float or numeric
    temp_set = test_set[test_set[emocja].apply(is_numeric_or_numeric_string)]
    # left
    temp_left =  test_set[~test_set[emocja].apply(is_numeric_or_numeric_string)]

    rejected.append(temp_left)
    # change datatype to int
    emo_llm = temp_set[emocja].values
    #  to int
    emo_llm = [float(str(e)) for e in emo_llm]
    if verbose:
        print(len(temp_set))
        print(np.corrcoef(emo_llm, temp_set[emotion]))
        # print standard deviations
        print(np.std(emo_llm))
        rejected_df = pd.concat(rejected)
        return rejected_df
    else:
        return np.corrcoef(emo_llm, temp_set[emotion]), np.std(emo_llm), len(temp_left)


emocje = ['valence', 'arousal']
emotions = ['norm_Valence_M', 'norm_Arousal_M']
two_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_two_shot_dimensions.csv')
two_shot_gpt4 = pd.read_csv('data/for_training/val_set_llm_gpt_4_0613_two_shot_val_dimensions.csv')

# create csv
types = ['two_shot', 'two_shot_gpt4']
correlation_list = []
standard_deviation_list = []
results = {}
for type, shot in zip(types, [two_shot, two_shot_gpt4]):
    # get mean correlation across emotions
    correlations = []
    standard_devs = []
    results[type] = {}
    for (emotion, emocja) in zip(emotions, emocje):
        correlation, Sd, rejected = check_correlation(shot, emotion, emocja, verbose = False)
        results[type][emocja] = {'correlation': correlation[0, 1], 'standard_deviation': Sd, 'rejected': rejected}


# save
collate = []
for type in types:
    for emo in emocje:
        collate.append({'type': type, 'emotion': emo, 'correlation': results[type][emo]['correlation'], 'standard_deviation': results[type][emo]['standard_deviation'], 'rejected': results[type][emo]['rejected']})

correlation_df_dim = pd.DataFrame(collate)

correlation_df_dim.to_csv('data/for_training/results_dimensions_final.csv', index=False)

# load
correlation_df_dim = pd.read_csv('data/for_training/results_dimensions_final.csv')

######## discrete
emocje = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie', 'Duma']
emotions = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Fear_M',
       'Disgust_M', 'Pride_M']

print('three shot')
# load
three_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_three_shot.csv')

print('three shot')
# load
three_shot_gpt4 = pd.read_csv('data/for_training/test_set_llm_gpt4_three_shot.csv')


types = ['three_shot', 'three_shot_gpt4']
correlation_list = []
standard_deviation_list = []
results = {}
for type, shot in zip(types, [three_shot, three_shot_gpt4]):
    # get mean correlation across emotions
    correlations = []
    standard_devs = []
    results[type] = {}
    for (emotion, emocja) in zip(emotions, emocje):
        correlation, Sd, rejected = check_correlation(shot, emotion, emocja, verbose = False)
        results[type][emocja] = {'correlation': correlation[0, 1], 'standard_deviation': Sd, 'rejected': rejected}


# save
collate = []
for type in types:
    for emo in emocje:
        collate.append({'type': type, 'emotion': emo, 'correlation': results[type][emo]['correlation'], 'standard_deviation': results[type][emo]['standard_deviation'], 'rejected': results[type][emo]['rejected']})

correlation_df_dis = pd.DataFrame(collate)
correlation_df_dis.to_csv('data/for_training/results_discrete_final.csv', index=False)

# load
correlation_df_dis = pd.read_csv('data/for_training/results_discrete_final.csv')
