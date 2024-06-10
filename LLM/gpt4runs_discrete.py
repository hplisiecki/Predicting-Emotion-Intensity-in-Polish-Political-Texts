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
### gpt-4-0613


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


    annotation_module = ActivationGPT(model="gpt-4", system_message=query, T=0, max_characters=12)

    for text in tqdm(test_set['text']):
        annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
        collated.append(str(annotation))

    return collated



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
test_set.to_csv('data/for_training/test_set_llm_gpt4_three_shot.csv', index=False)
