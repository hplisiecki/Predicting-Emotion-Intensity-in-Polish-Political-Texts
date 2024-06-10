import tiktoken
import pandas as pd
import json
from tqdm import tqdm
api_file = r"C:\Users\hplis\OneDrive\Desktop\PHD\OPENAI\api_keys.json"

# load api key
with open(api_file) as f:
    api_key = json.load(f)




emotions = ['Szczęście', 'Smutek', 'Gniew', 'Obrzydzenie', 'Strach', 'Duma']
emotion_columns = ['norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M']
# BASIC EMOTIONS
def multiple_shot(val_set, emotion, examples = [], values = [], debug = False):


    collated = []
    idx = 0

    query = (
        f'Na ile przedstawiony poniżej tekst manifestuje emocje "{emotion}". Odpowiedz używając 5 stopniowej skali, '
        f'gdzie 1 - emocja wogóle nie występuje a 5 - emocja jest bardzo wyraźnie obecna. Odpowiadaj za pomocą '
        f'pojedynczego numeru. ')


    for idx, tuple in enumerate(zip(examples, values)):
        query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'


    next_text_idx = idx

    for text in tqdm(val_set['text']):
        collated.append(query + f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: ')

    return collated

######################################################################
########################## ZEROSHOT ##################################
######################################################################
val_set = pd.read_csv('data/for_training/new_training_data/val_set.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_zero_shot_val.csv')

queries_and_results = pd.DataFrame()

for emotion in emotions:
    queries = multiple_shot(val_set, emotion)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_zero_shot_val_queries.csv', index=False)

######################################################################
########################## ONESHOT ###################################
######################################################################
one_shot = pd.read_csv('data/embeddings/one_shot.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_one_shot_val.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(one_shot[one_shot['emotion'] == emotion_column]['text'].values)
    values = list(one_shot[one_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_one_shot_val_queries.csv', index=False)


######################################################################
########################## TWOSHOT ###################################
######################################################################
two_shot = pd.read_csv('data/embeddings/two_shot.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_two_shot_val.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_two_shot_val_queries.csv', index=False)

######################################################################
########################## THREESHOT #################################
######################################################################

one_shot = pd.read_csv('data/embeddings/one_shot.csv')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_three_shot_val.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    one_shot_exemplar = one_shot[one_shot['emotion'] == emotion_column]['text'].iloc[0]
    one_shot_value = one_shot[one_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].iloc[0]
    exemplars.append(one_shot_exemplar)
    values.append(one_shot_value)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_three_shot_val_queries.csv', index=False)

######################################################################
########################## FOURSHOT  four points #####################
######################################################################

four_shot = pd.read_csv('data/embeddings/four_shot_four_points.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_four_shot_four_points_val.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(four_shot[four_shot['emotion'] == emotion_column]['text'].values)
    values = list(four_shot[four_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_four_shot_four_points_val_queries.csv', index=False)


######################################################################
########################## FIVESHOT  five points #####################
######################################################################

five_shot = pd.read_csv('data/embeddings/five_shot_five_points.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_five_shot_five_points_val.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(five_shot[five_shot['emotion'] == emotion_column]['text'].values)
    values = list(five_shot[five_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_five_shot_five_points_val_queries.csv', index=False)


######################################################################
############################ TEST RUNS ###############################
######################################################################

test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')

results_gpt4 = pd.read_csv('data/for_training/test_set_llm_gpt4_three_shot.csv')
results_gpt3 = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_three_shot.csv')
queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_gpt4_result'] = results_gpt4[emotion]
    queries_and_results[emotion + '_gpt3_result'] = results_gpt3[emotion]

queries_and_results.to_csv('data/for_training/test_runs_discrete_queries.csv', index=False)



########################## CALCULATING COST ########################################
import tiktoken
emotions = ['Szczęście', 'Smutek', 'Gniew', 'Obrzydzenie', 'Strach', 'Duma']

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# all_texts = ' '.join(val_set['text'].values)

# num_tokens_from_string(all_texts, "cl100k_base")

all_query_tokens = 0
all_response_tokens = 0

# zeroshot
queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_zero_shot_val_queries.csv', index=False)
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_responses = queries[emotion + '_result']
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# oneshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_one_shot_val_queries.csv')

for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_responses = queries[emotion + '_result']
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# twoshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_two_shot_val_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_responses = queries[emotion + '_result']
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# threeshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_three_shot_val_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_responses = queries[emotion + '_result']
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# fourshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_four_shot_four_points_val_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_responses = queries[emotion + '_result']
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# fiveshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_five_shot_five_points_val_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_responses = queries[emotion + '_result']
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

print(all_query_tokens)
print(all_response_tokens)

# 0.5 per 1M tokens
query_tokens_price = all_query_tokens * 0.5 / 1000000
# 1.5 per 1M tokens
response_tokens_price = all_response_tokens * 1.5 / 1000000
total_discrete_grid_search_price = query_tokens_price + response_tokens_price
print(total_discrete_grid_search_price) # 6.5076045 $


# test runs
queries = pd.read_csv('data/for_training/test_runs_discrete_queries.csv')

all_gpt3_query_tokens = 0
all_gpt4_query_tokens = 0
all_gpt3_response_tokens = 0
all_gpt4_response_tokens = 0
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")

    gpt3_responses = queries[emotion + '_gpt3_result']
    all_texts = ' '.join(gpt3_responses)
    response_tokens_gpt3 = num_tokens_from_string(all_texts, "cl100k_base")

    gpt4_responses = queries[emotion + '_gpt4_result']
    all_texts = ' '.join(gpt4_responses)
    response_tokens_gpt4 = num_tokens_from_string(all_texts, "cl100k_base")

    all_gpt3_query_tokens += query_tokens
    all_gpt4_query_tokens += query_tokens
    all_gpt3_response_tokens += response_tokens_gpt3
    all_gpt4_response_tokens += response_tokens_gpt4

# 0.5 per 1M tokens
query_gpt3_price = all_gpt3_query_tokens * 0.5 / 1000000
# 1.5 per 1M tokens
response_gpt3_price = all_gpt3_response_tokens * 1.5 / 1000000

# 30 per 1M tokens
query_gpt4_price = all_gpt4_query_tokens * 30 / 1000000

# 60 per 1M tokens
response_gpt4_price = all_gpt4_response_tokens * 60 / 1000000

total_price = query_gpt3_price + response_gpt3_price + query_gpt4_price + response_gpt4_price
print(total_price) # 49.23 $













#################################################################################################################
#################################################################################################################
#################################################################################################################
############################################ VALENCE AROUSAL ####################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

emotions = ['valence', 'arousal']
emotion_columns = ['norm_Valence_M', 'norm_Arousal_M']
def multiple_shot(val_set, emotion, examples = [], values = [], debug = False):


    collated = []
    idx = 0

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

    for idx, tuple in enumerate(zip(examples, values)):
        query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'


    next_text_idx = idx

    for text in tqdm(val_set['text']):
        collated.append(query + f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: ')

    return collated


######################################################################
########################## ZEROSHOT ##################################
######################################################################

val_set = pd.read_csv('data/for_training/new_training_data/val_set.csv')

results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_zero_shot_val_dimensions.csv')
queries_and_results = pd.DataFrame()

for emotion in emotions:
    queries = multiple_shot(val_set, emotion)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_zero_shot_val_dimensions_queries.csv', index=False)


######################################################################
########################## ONESHOT ###################################
######################################################################
one_shot = pd.read_csv('data/embeddings/one_shot.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_one_shot_val_dimensions.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(one_shot[one_shot['emotion'] == emotion_column]['text'].values)
    values = list(one_shot[one_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_one_shot_val_dimensions_queries.csv', index=False)


######################################################################
########################## TWOSHOT ###################################
######################################################################

two_shot = pd.read_csv('data/embeddings/two_shot.csv')
results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_two_shot_val_dimensions.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_two_shot_val_dimensions_queries.csv', index=False)


######################################################################
########################## THREESHOT #################################
######################################################################

one_shot = pd.read_csv('data/embeddings/one_shot.csv')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')

results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_three_shot_val_dimensions.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    one_shot_exemplar = one_shot[one_shot['emotion'] == emotion_column]['text'].iloc[0]
    one_shot_value = one_shot[one_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].iloc[0]
    exemplars.append(one_shot_exemplar)
    values.append(one_shot_value)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_three_shot_val_dimensions_queries.csv', index=False)

######################################################################
########################## FOURSHOT  four points #####################
######################################################################
four_shot = pd.read_csv('data/embeddings/four_shot_four_points.csv')

results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_four_shot_four_points_val_dimensions.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(four_shot[four_shot['emotion'] == emotion_column]['text'].values)
    values = list(four_shot[four_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_four_shot_four_points_val_dimensions_queries.csv', index=False)

######################################################################
########################## FIVESHOT  five points #####################
######################################################################

five_shot = pd.read_csv('data/embeddings/five_shot_five_points.csv')

results = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_five_shot_five_points_val_dimensions.csv')

queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(five_shot[five_shot['emotion'] == emotion_column]['text'].values)
    values = list(five_shot[five_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_result'] = results[emotion]

queries_and_results.to_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_five_shot_five_points_val_dimensions_queries.csv', index=False)


######################################################################
############################ TEST RUNS ###############################
######################################################################

test_set = pd.read_csv('data/for_training/new_training_data/test_set.csv')
two_shot = pd.read_csv('data/embeddings/two_shot.csv')

results_gpt4 = pd.read_csv('data/for_training/test_set_llm_gpt4_two_shot_dimensions.csv')
results_gpt3 = pd.read_csv('data/for_training/test_set_llm_gpt3_5_turbo_0125_two_shot_dimensions.csv')
queries_and_results = pd.DataFrame()

for emotion, emotion_column in zip(emotions, emotion_columns):
    exemplars = list(two_shot[two_shot['emotion'] == emotion_column]['text'].values)
    values = list(two_shot[two_shot['emotion'] == emotion_column][emotion_column.replace('norm_', '')].values)
    queries = multiple_shot(val_set, emotion, examples = exemplars, values = values)
    queries_and_results[emotion + '_query'] = queries
    queries_and_results[emotion + '_gpt4_result'] = results_gpt4[emotion]
    queries_and_results[emotion + '_gpt3_result'] = results_gpt3[emotion]


# save
queries_and_results.to_csv('data/for_training/test_runs_dimensions_queries.csv', index=False)


########################## CALCULATING COST ########################################
import tiktoken
emotions = ['valence', 'arousal']
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

all_query_tokens = 0
all_response_tokens = 0

# zeroshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_zero_shot_val_dimensions_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    # as string
    all_responses = queries[emotion + '_result'].apply(str)
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens


# oneshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_one_shot_val_dimensions_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    # as string
    all_responses = queries[emotion + '_result'].apply(str)
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# twoshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_two_shot_val_dimensions_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    # as string
    all_responses = queries[emotion + '_result'].apply(str)
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# threeshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_three_shot_val_dimensions_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    # as string
    all_responses = queries[emotion + '_result'].apply(str)
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# fourshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_four_shot_four_points_val_dimensions_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    # as string
    all_responses = queries[emotion + '_result'].apply(str)
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

# fiveshot
queries = pd.read_csv('data/for_training/val_set_llm_gpt3_5_turbo_0125_five_shot_five_points_val_dimensions_queries.csv')
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    # as string
    all_responses = queries[emotion + '_result'].apply(str)
    all_texts = ' '.join(all_responses)
    response_tokens = num_tokens_from_string(all_texts, "cl100k_base")
    all_query_tokens += query_tokens
    all_response_tokens += response_tokens

print(all_query_tokens)
print(all_response_tokens)

# 0.5 per 1M tokens
query_tokens_price = all_query_tokens * 0.5 / 1000000
# 1.5 per 1M tokens
response_tokens_price = all_response_tokens * 1.5 / 1000000
total_dimension_grid_search_price = query_tokens_price + response_tokens_price
print(total_dimension_grid_search_price) # 1.874829 $



# test runs
queries = pd.read_csv('data/for_training/test_runs_dimensions_queries.csv')

all_gpt3_query_tokens = 0
all_gpt4_query_tokens = 0
all_gpt3_response_tokens = 0
all_gpt4_response_tokens = 0
for emotion in emotions:
    all_queries = queries[emotion + '_query']
    all_texts = ' '.join(all_queries)
    query_tokens = num_tokens_from_string(all_texts, "cl100k_base")

    gpt3_responses = queries[emotion + '_gpt3_result']
    all_texts = ' '.join(gpt3_responses)
    response_tokens_gpt3 = num_tokens_from_string(all_texts, "cl100k_base")

    gpt4_responses = queries[emotion + '_gpt4_result']
    all_texts = ' '.join(gpt4_responses)
    response_tokens_gpt4 = num_tokens_from_string(all_texts, "cl100k_base")

    all_gpt3_query_tokens += query_tokens
    all_gpt4_query_tokens += query_tokens
    all_gpt3_response_tokens += response_tokens_gpt3
    all_gpt4_response_tokens += response_tokens_gpt4

# 0.5 per 1M tokens
query_gpt3_price = all_gpt3_query_tokens * 0.5 / 1000000
# 1.5 per 1M tokens
response_gpt3_price = all_gpt3_response_tokens * 1.5 / 1000000

# 30 per 1M tokens
query_gpt4_price = all_gpt4_query_tokens * 30 / 1000000

# 60 per 1M tokens
response_gpt4_price = all_gpt4_response_tokens * 60 / 1000000

total_price = query_gpt3_price + response_gpt3_price + query_gpt4_price + response_gpt4_price
print(total_price) # 16.37 $
