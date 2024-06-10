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

def multiple_shot(val_set, emotion, examples = [], values = [], debug = False):


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

    for text in tqdm(val_set['text']):
        annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
        collated.append(str(annotation))

    return collated
