import tiktoken
import pandas as pd
import json
api_file = r"C:\Users\hplis\OneDrive\Desktop\PHD\OPENAI\api_keys.json"

# load api key
with open(api_file) as f:
    api_key = json.load(f)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

train_set = pd.read_csv('data/for_training/new_training_data/train_set.csv')

all_texts = ' '.join(train_set['text'].values)

num_tokens_from_string(all_texts, "cl100k_base")




from openai import OpenAI
client = OpenAI(api_key=api_key['openai_api_key'])

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


train_set['embeddings'] = train_set.text.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
train_set['embeddings'] = train_set['embeddings'].apply(json.dumps)
# save
train_set.to_csv('data/embeddings/train_set.csv', index=False)
# load
df = pd.read_csv('data/embeddings/train_set.csv')
df['ada_embedding'] = df['ada_embedding'].apply(json.loads)
df.columns = ['text', 'source', 'part', 'Happiness_M', 'Sadness_M', 'Anger_M',
       'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M', 'Irony_M',
       'norm_Happiness_M', 'norm_Sadness_M', 'norm_Anger_M', 'norm_Disgust_M',
       'norm_Fear_M', 'norm_Pride_M', 'norm_Valence_M', 'norm_Arousal_M',
       'z_score_Happiness_M', 'z_score_Sadness_M', 'z_score_Anger_M',
       'z_score_Disgust_M', 'z_score_Fear_M', 'z_score_Pride_M',
       'z_score_Valence_M', 'z_score_Arousal_M', 'total_z_score',
       'embeddings']

# serialize
df['embeddings'] = df['embeddings'].apply(json.dumps)
df.to_csv('data/embeddings/train_set.csv', index=False)

