import sys
from tqdm import tqdm

# sys.path.append(r'D:\GitHub\LangTorch\src')


from langtorch.tt import ActivationGPT, TextModule
from langtorch import TextTensor
from langtorch.api import auth
from tqdm import tqdm
import pandas as pd
import numpy as np
api_file = r"C:\Users\hplis\OneDrive\Desktop\PHD\OPENAI\api_keys.json"

auth(api_file)

emotions = ['Szczęście', 'Smutek', 'Gniew', 'Obrzydzenie', 'Strach', 'Duma']
emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M']

def multiple_shot(test_set, emotion, examples = [], values = [], debug = False):


    collated = []
    idx = 0

    query = (
        f'Na ile przedstawiony poniżej tekst manifestuje emocje "{emotion}". Odpowiedz używając 5 stopniowej skali, '
        f'gdzie 1 - emocja wogóle nie występuje a 5 - emocja jest bardzo wyraźnie obecna. Odpowiadaj za pomocą '
        f'pojedynczego numeru. ')


    for idx, tuple in enumerate(zip(examples, values)):
        query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'

    if debug:
        print(query)
        return

    next_text_idx = idx


    annotation_module = ActivationGPT(model="gpt-3.5-turbo-1106", system_message=query, T=0, max_characters=12)

    for text in tqdm(test_set['text']):
        annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
        collated.append(str(annotation))

    return collated


######################################################################
########################## ZEROSHOT ##################################
######################################################################
print('Zeroshot')
test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')

for emotion in emotions:
    print(emotion)
    test_set[emotion] = multiple_shot(test_set, emotion)
test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_zero_shot.csv', index=False)

######################################################################
########################## ONESHOT ###################################
######################################################################

print('Oneshot')
one_shot = pd.read_csv('data/embeddings/one_shot.csv')
test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
for emotion, emotion_column in zip(emotions, emotion_columns):
    print(emotion)
    exemplar = one_shot[one_shot['emotion'] == emotion_column]['text'].iloc[0]
    value = one_shot[one_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].iloc[0]
    test_set[emotion] = multiple_shot(test_set, emotion, examples = [exemplar], values = [value])

test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_one_shot.csv', index=False)


######################################################################
########################## TWOSHOT ###################################
######################################################################


print('Twoshot')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')
test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
for emotion, emotion_column in zip(emotions, emotion_columns):
    print(emotion)
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    test_set[emotion] = multiple_shot(test_set, emotion, examples = exemplars, values = values)

# save
test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_two_shot.csv', index=False)



######################################################################
########################## THREESHOT #################################
######################################################################


print('Three shot')
one_shot = pd.read_csv('data/embeddings/one_shot.csv')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')
test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
for emotion, emotion_column in zip(emotions, emotion_columns):
    print(emotion)
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    one_shot_exemplar = one_shot[one_shot['emotion'] == emotion_column]['text'].iloc[0]
    one_shot_value = one_shot[one_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].iloc[0]
    exemplars.append(one_shot_exemplar)
    values.append(one_shot_value)
    test_set[emotion] = multiple_shot(test_set, emotion, examples = exemplars, values = values)

# save
test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_three_shot.csv', index=False)


######################################################################
########################## FOURSHOT  four points #####################
######################################################################

print('Four shot four points')
# load
four_shot = pd.read_csv('data/embeddings/four_shot_four_points.csv')
test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
for emotion, emotion_column in zip(emotions, emotion_columns):
    print(emotion)
    exemplars = list(four_shot[four_shot['emotion'] == emotion_column]['text'].values)
    values = list(four_shot[four_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    test_set[emotion] = multiple_shot(test_set, emotion, examples = exemplars, values = values)

# save
test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_four_shot_four_points.csv', index=False)

######################################################################
########################## FIVESHOT  five points #####################
######################################################################

print('Five shot four points')
# load
five_shot = pd.read_csv('data/embeddings/five_shot_five_points.csv')
test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
for emotion, emotion_column in zip(emotions, emotion_columns):
    print(emotion)
    exemplars = list(five_shot[five_shot['emotion'] == emotion_column]['text'].values)
    values = list(five_shot[five_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    test_set[emotion] = multiple_shot(test_set, emotion, examples = exemplars, values = values)

# save
test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_five_shot_five_points.csv', index=False)



######################################################################
############################# TESTING ################################
######################################################################


import pandas as pd
import numpy as np
emocje = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie']
emotions = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Fear_M',
       'Disgust_M']
def is_numeric_or_numeric_string(x):
    try:
        float(x)  # Attempt to convert to float
        return True  # Conversion successful
    except ValueError:
        return False  # Conversion failed, value is not numeric

def check_correlation(test_set, verbose = True):
    if verbose:
        print("#########################################")
    rejected = []
    for (emotion, emocja) in zip(emotions, emocje):
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
        else:
            return np.corrcoef(emo_llm, temp_set[emotion]), np.std(emo_llm)

    rejected_df = pd.concat(rejected)
    if verbose:
        return rejected_df


import pandas as pd
print('zero shot')
# load
zero_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_zero_shot.csv')
left_zero = check_correlation(zero_shot)

print('one shot')
# load
one_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_one_shot.csv')
left_one = check_correlation(one_shot)

print('two shot')
# load
two_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_two_shot.csv')
left_two = check_correlation(two_shot)

print('three shot')
# load
three_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_three_shot.csv')
left_three = check_correlation(three_shot)

print('four shot four points')
# load
four_shot_four_points = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_four_shot_four_points.csv')
left_four_points = check_correlation(four_shot_four_points)

print('five shot five points')
# load
five_shot_five_points = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_five_shot_five_points.csv')
left_five_points = check_correlation(five_shot_five_points)



types = ['zero_shot', 'one_shot', 'two_shot', 'three_shot', 'four_shot_four_points', 'five_shot_five_points']
correlation_list = []
standard_deviation_list = []
for type, shot in zip(types, [zero_shot, one_shot, two_shot, three_shot, four_shot_four_points, five_shot_five_points]):
    # get mean correlation across emotions
    correlations = []
    standard_devs = []
    for emo in emotions:
        correlation, Sd = check_correlation(shot, verbose = False)
        correlations.append(correlation[0, 1])
        standard_devs.append(Sd)
    correlation_list.append(np.mean(correlations))
    standard_deviation_list.append(np.mean(standard_devs))

# save
correlation_df = pd.DataFrame({'type': types, 'correlation': correlation_list, 'standard_deviation': standard_deviation_list})
correlation_df.to_csv('data/for_training/results_test.csv', index=False)