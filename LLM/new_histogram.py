import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Valence and arousal
emocje_dim = ['valence', 'arousal']
# Discrete emotions
emocje_dis = ['Szczęście', 'Smutek', 'Gniew', 'Strach', 'Obrzydzenie', 'Duma']

annotator_emotions = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Pride', 'Valence', 'Arousal']

# Example discrete data

two_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_two_shot_dimensions.csv')
two_shot_gpt4 = pd.read_csv('data/for_training/val_set_llm_gpt_4_0613_two_shot_val_dimensions.csv')
three_shot = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_three_shot.csv')
three_shot_gpt4 = pd.read_csv('data/for_training/test_set_llm_gpt4_three_shot.csv')

import pandas as pd
annotators_results = pd.read_csv('data/for_training/wyniki_dla_piotra.csv')
annotators_results = annotators_results[annotators_results['text'].isin(two_shot['text'])]
annotators_results = annotators_results[annotator_emotions]

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
df_gpt3_5 = pd.concat([three_shot, two_shot], axis=1)
df_gpt4 = pd.concat([three_shot_gpt4, two_shot_gpt4], axis=1)

df_gpt3_5 = df_gpt3_5[emocje_dis + emocje_dim]
df_gpt4 = df_gpt4[emocje_dis + emocje_dim]

df_gpt3_5.columns = annotator_emotions
df_gpt4.columns = annotator_emotions

def is_numeric_or_numeric_string(x):
    try:
        float(x)  # Attempt to convert to float
        return True  # Conversion successful
    except ValueError:
        return False  # Conversion failed, value is not numeric

def clean_column(df, column):
    df_valid = df[df[column].apply(is_numeric_or_numeric_string)]
    df_valid[column] = df_valid[column].astype(float)
    return df_valid[column]


# Function to plot side-by-side histograms with normalization for one dataset
def plot_side_by_side_histograms(ax, data_list, labels, colors, hatches, width=0.2, normalize_index=None, normalize_factor=1):
    x = np.arange(1, 6)  # Define x values for the discrete bins
    offsets = np.array([-width, 0, width])  # Adjust offsets to center the green bin

    for i, (data, label, color, hatch, offset) in enumerate(zip(data_list, labels, colors, hatches, offsets)):
        counts, _ = np.histogram(data, bins=np.arange(1, 7))
        if i == normalize_index:
            counts = counts / normalize_factor  # Normalize the specified data source
        ax.bar(x + offset, counts, width=width, label=label, color=color, hatch=hatch, align='center')

    ax.legend(frameon=False, loc='upper right')



# Creating a grid of subplots for each emotion column
fig, axs = plt.subplots(4, 2, figsize=(8.27, 9.69), constrained_layout=True)  # Reduced height for additional space on A4 page
axs = axs.flatten()

# Adjusted order of labels and colors
labels = ['Annotators', 'GPT-3.5', 'GPT-4']
colors = ['red', 'blue', 'green']
hatches = ['////', '....', 'xxxx']  # Denser hatch patterns for each dataset

# Set APA style parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8
})

# Plotting the histograms for each emotion
for i, emotion in enumerate(annotator_emotions):
    data_list = [
        annotators_results[emotion].dropna(),  # Assuming annotators_results is already clean
        clean_column(df_gpt3_5, emotion).dropna(),
        clean_column(df_gpt4, emotion).dropna()
    ]
    plot_side_by_side_histograms(axs[i], data_list, labels, colors, hatches, normalize_index=0, normalize_factor=5)
    axs[i].set_title(emotion, fontsize=14, fontname='Times New Roman')
    axs[i].set_xticks(np.arange(1, 6))  # Adjust this based on the range of your data
    axs[i].set_xticklabels(np.arange(1, 6), fontname='Times New Roman')
    axs[i].set_xlabel('Rating', fontsize=10, fontname='Times New Roman')
    axs[i].set_ylabel('Frequency', fontsize=10, fontname='Times New Roman')
    axs[i].tick_params(axis='y', which='major', labelsize=10)
    for label in axs[i].get_yticklabels():
        label.set_fontname('Times New Roman')


# Adjust layout to add more vertical space between rows
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.5)  # Increase vertical space between rows

# Save the plot as a JPEG file
plt.savefig('plots/histograms_A4.jpeg', format='jpeg', dpi=300)

# Display the plot
plt.show()