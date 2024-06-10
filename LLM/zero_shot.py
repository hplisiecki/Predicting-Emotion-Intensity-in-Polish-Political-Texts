import sys
from tqdm import tqdm

sys.path.append(r'D:\GitHub\LangTorch\src')

from langtorch.tt import ActivationGPT, TextModule
from langtorch import TextTensor
from openapi import auth
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
api_file = r"C:\Users\hplis\OneDrive\Desktop\PHD\OPENAI\api_keys.json"

auth(key_path=api_file)

test_set = pd.read_csv('data/for_training/new_training_data/val_set.csv')

emotion = 'Szczęście'


text = test_set['text'].iloc[0]



emotions = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie']
for emotion in emotions:
    print(emotion)
    collated = []
    annotation_module = ActivationGPT( (f'Na ile przedstawiony poniżej tekst manifestuje emocje "{emotion}". Odpowiedz używając 5 stopniowej skali, '
                                        f'gdzie 1 - emocja wogóle nie występuje a 5 - emocja jest bardzo wyraźnie obecna. Odpowiadaj za pomocą '
                                        f'pojedynczego numeru. Tekst: """{text}"" Twoja odpowiedź: '),
                                      model="gpt-3.5-turbo-1106", temperature=0, max_characters=12)

    for text in tqdm(test_set['text']):
        annotation = annotation_module(TextTensor(text))
        collated.append(annotation)

    test_set[emotion] = collated


# save
test_set.to_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_zero_shot_val.csv', index=False)

# load
test_set = pd.read_csv('data/for_training/new_training_data/test_set_llm_gpt3_5_turbo_0125.csv')




print('GPT 3.5')

import numpy as np
emocje = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie']
emotions = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Fear_M',
       'Disgust_M']
for (emotion, emocja) in zip(emotions, emocje):
    # delete nonnumeric
    print(emocja)
    temp_set = test_set[test_set[emocja].apply(lambda x: x.isnumeric())]
    # change datatype to int
    emo_llm = temp_set[emocja].values
    #  to int
    emo_llm = [int(str(e)) for e in emo_llm]
    print(len(temp_set))
    print(np.corrcoef(emo_llm, temp_set[emotion]))








print('#########################################')

print('GPT 4')
test_set = pd.read_csv('data/for_training/new_training_data/test_set_llm_gpt3_5_turbo_0125.csv')


import numpy as np
emocje = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie']
emotions = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Fear_M',
       'Disgust_M']
for (emotion, emocja) in zip(emotions, emocje):
    # delete nonnumeric
    print(emocja)
    temp_set = test_set[test_set[emocja].apply(lambda x: str(x).isnumeric())]
    # change datatype to int
    emo_llm = temp_set[emocja].values
    #  to int
    emo_llm = [int(str(e)) for e in emo_llm]
    print(len(temp_set))
    print(np.corrcoef(emo_llm, temp_set[emotion]))




with open(os.path.join(ANNOTATIONS_DIR, 'results_all_corr.pkl'), 'wb') as file:
    pickle.dump(results_all_corr, file)

with open(os.path.join(ANNOTATIONS_DIR, 'results_all_pred_mean.pkl'), 'wb') as file:
    pickle.dump(results_all_pred_mean, file)

with open(os.path.join(ANNOTATIONS_DIR, 'results_all_real_mean.pkl'), 'wb') as file:
    pickle.dump(results_all_real_mean, file)


# load all three

with open(os.path.join(ANNOTATIONS_DIR, 'results_all_corr.pkl'), 'rb') as file:
    results_all_corr = pickle.load(file)

with open(os.path.join(ANNOTATIONS_DIR, 'results_all_pred_mean.pkl'), 'rb') as file:
    results_all_pred_mean = pickle.load(file)

with open(os.path.join(ANNOTATIONS_DIR, 'results_all_real_mean.pkl'), 'rb') as file:
    results_all_real_mean = pickle.load(file)