import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def is_numeric_or_numeric_string(x):
    try:
        float(x)  # Attempt to convert to float
        return True  # Conversion successful
    except ValueError:
        return False  # Conversion failed, value is not numeric


def generate_histograms(df, emotions, emocje, title, save_path):
    num_emotions = len(emotions)
    cols = 3
    rows = (num_emotions // cols) + (num_emotions % cols > 0)

    plt.figure(figsize=(20, 15))

    for i, (emotion, emocja) in enumerate(zip(emotions, emocje)):
        plt.subplot(rows, cols, i + 1)
        df_valid = df[df[emocja].apply(is_numeric_or_numeric_string)]
        df_valid[emocja] = df_valid[emocja].astype(float)

        plt.hist(df_valid[emocja], bins=30, alpha=0.5, label=f'Predicted {emocja}', color='blue')
        plt.hist(df_valid[emotion], bins=30, alpha=0.5, label=f'Real {emotion}', color='orange')

        plt.xlabel(f'{emocja}')
        plt.ylabel('Frequency')
        plt.title(f'{emocja}')
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    # close
    plt.close()




# Load the data
two_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_two_shot_dimensions.csv')
two_shot_gpt4 = pd.read_csv('data/for_training/val_set_llm_gpt_4_0613_two_shot_val_dimensions.csv')
three_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_three_shot.csv')
three_shot_gpt4 = pd.read_csv('data/for_training/test_set_llm_gpt4_three_shot.csv')


import pandas as pd
df = pd.read_csv('data/for_training/wyniki_dla_piotra.csv')
df = df[df['text'].isin(two_shot['text'])]


# Valence and arousal
emocje_dim = ['valence', 'arousal']
emotions_dim = ['Valence_M', 'Arousal_M']

# Discrete emotions
emocje_dis = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie', 'Duma']
emotions_dis = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Fear_M', 'Disgust_M', 'Pride_M']

# cleaning
for emo in emocje_dis:
    three_shot[emo] = three_shot[emo].apply(lambda x: str(x).replace('"', ''))
    three_shot_gpt4[emo] = three_shot_gpt4[emo].apply(lambda x: str(x).replace('"', ''))


for emo in emocje_dim:
    two_shot[emo] = two_shot[emo].apply(lambda x: str(x).replace('"', ''))
    two_shot_gpt4[emo] = two_shot_gpt4[emo].apply(lambda x: str(x).replace('"', ''))

two_shot_gpt4 = two_shot_gpt4[emocje_dim]
two_shot = two_shot[emocje_dim]

# Merge the dataframes for GPT-3.5 and GPT-4
df_gpt3_5 = pd.concat([two_shot, three_shot], axis=1)
df_gpt4 = pd.concat([two_shot_gpt4, three_shot_gpt4], axis=1)

plot_path = 'plots/'

# Generate histograms for GPT-3.5 and GPT-4
generate_histograms(df_gpt3_5, emotions_dim + emotions_dis, emocje_dim + emocje_dis, 'GPT-3.5 Emotions', os.path.join(plot_path, 'GPT3_5_histograms.png'))
generate_histograms(df_gpt4, emotions_dim + emotions_dis, emocje_dim + emocje_dis, 'GPT-4 Emotions', os.path.join(plot_path, 'GPT4_histograms.png'))




def generate_scatterplots(df, emotions, emocje, title, save_path):
    num_emotions = len(emotions)
    cols = 3
    rows = (num_emotions // cols) + (num_emotions % cols > 0)

    plt.figure(figsize=(20, 15))
    for i, (emotion, emocja) in enumerate(zip(emotions, emocje)):
        plt.subplot(rows, cols, i + 1)
        df_valid = df[df[emocja].apply(is_numeric_or_numeric_string)]
        df_valid[emocja] = df_valid[emocja].astype(float)

        plt.scatter(df_valid[emotion], df_valid[emocja], alpha=0.5, label=f'{emocja} vs {emotion}', color='blue')

        plt.xlabel(f'Real {emotion}')
        plt.ylabel(f'Predicted {emocja}')
        plt.title(f'{emocja} vs {emotion}')
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    # close
    plt.close()

# Generate scatterplots for GPT-3.5 and GPT-4
generate_scatterplots(df_gpt3_5, emotions_dim + emotions_dis, emocje_dim + emocje_dis, 'GPT-3.5 Emotions', os.path.join(plot_path, 'GPT3_5_scatterplots.png'))
generate_scatterplots(df_gpt4, emotions_dim + emotions_dis, emocje_dim + emocje_dis, 'GPT-4 Emotions', os.path.join(plot_path, 'GPT4_scatterplots.png'))

