#############################
#### VALENCE AND AROUSAL ####
#############################
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


emotions = ['valence', 'arousal']
emotion_columns = ['norm_Valence_M', 'norm_Arousal_M']

def multiple_shot(val_set, emotion, examples = [], values = [], debug = False):


    if emotion == 'valence':
        query = (
            f'Jaki znak emocji wyczytujesz w poniższym tekście? Odpowiedz używając 5 stopniowej skali, '
            f'gdzie 1 - obecna jest negatywna emocja a 5 - obecna jest pozytywna emocja. Odpowiadaj za pomocą '
            f'pojedynczego numeru. ')

    if emotion == 'arousal':
        query = (
            f'Jaki poziom pobudzenia wyczytujesz w poniższym tekście? Odpowiedz używając 5 stopniowej skali, '
            f'gdzie 1 - brak pobudzenia a 5 - ekstremalne pobudzenie. Odpowiadaj za pomocą '
            f'pojedynczego numeru. ')

    collated = []
    idx = 0




    for idx, tuple in enumerate(zip(examples, values)):
        query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'

    if debug:
        print(query)
        return

    next_text_idx = idx


    annotation_module = ActivationGPT(model="gpt-3.5-turbo-1106", system_message=query, T=0, max_characters=12)

    for text in tqdm(val_set['text']):
        annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
        collated.append(str(annotation))

    return collated


######################################################################
########################## TWOSHOT ###################################
######################################################################


print('Twoshot')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')
val_set = pd.read_csv('data/for_training/new_training_data/val_set.csv')

emotion = 'valence'
emotion_column = 'norm_Valence_M'
exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)

test_set = val_set.copy()

examples = exemplars
collated = []
idx = 0


# VALENCE QUERY
query = (
    f'Jaki znak emocji wyczytujesz w poniższym tekście? Odpowiedz używając 5 stopniowej skali, '
    f'gdzie 1 - obecna jest negatywna emocja a 5 - obecna jest pozytywna emocja. Odpowiadaj za pomocą '
    f'pojedynczego numeru. ')


for idx, tuple in enumerate(zip(examples, values)):
    query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'


next_text_idx = idx


annotation_module = ActivationGPT(model="gpt-4", system_message=query, T=0, max_characters=12)

for text in tqdm(test_set['text']):
    annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
    collated.append(str(annotation))

val_set[emotion] = collated

val_set.to_csv('data/for_training/val_set_llm_gpt_4_0613_two_shot_val_dimensions.csv', index=False)

emotion = 'arousal'
emotion_column = 'norm_Arousal_M'
exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)


test_set = val_set.copy()

examples = exemplars
collated = []
idx = 0

# AROUSAL QUERY
query = (
    f'Jaki poziom pobudzenia wyczytujesz w poniższym tekście? Odpowiedz używając 5 stopniowej skali, '
    f'gdzie 1 - brak pobudzenia a 5 - ekstremalne pobudzenie. Odpowiadaj za pomocą '
    f'pojedynczego numeru. ')

for idx, tuple in enumerate(zip(examples, values)):
    query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'


next_text_idx = idx


annotation_module = ActivationGPT(model="gpt-4", system_message=query, T=0, max_characters=12)

for text in tqdm(test_set['text']):
    annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
    collated.append(str(annotation))

val_set[emotion] = collated
# save
val_set.to_csv('data/for_training/val_set_llm_gpt_4_0613_two_shot_val_dimensions.csv', index=False)


# TEST
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
test_set.to_csv('data/for_training/test_set_llm_gpt4_two_shot_dimensions.csv', index=False)





######################################################################
############################# TESTING ################################
######################################################################


import pandas as pd
import numpy as np
emocje = ['valence', 'arousal']
emotions = ['norm_Valence_M', 'norm_Arousal_M']
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


print('two shot')
# load
two_shot = pd.read_csv('data/for_training/val_set_llm_gpt_4_0613_two_shot_val_dimensions.csv')
left_two = check_correlation(two_shot)