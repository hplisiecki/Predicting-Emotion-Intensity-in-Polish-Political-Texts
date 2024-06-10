import pandas as pd


def normalize_column(df, columns):
    for column in columns:
        df['norm_' + column] = (df[column] - min(df[column])) / (max(df[column]) - min(df[column]))
    return df


def load_data(emotion_columns):
    train = pd.read_csv('data/for_training/train_set.csv')
    val = pd.read_csv('data/for_training/val_set.csv')
    test = pd.read_csv('data/for_training/test_set.csv')

    train = train.dropna(subset=emotion_columns)
    val = val.dropna(subset=emotion_columns)
    test = test.dropna(subset=emotion_columns)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train = normalize_column(train, emotion_columns)
    val = normalize_column(val, emotion_columns)
    test = normalize_column(test, emotion_columns)

    return train, val, test


def check_max_token_length(tokenizer):
    train = pd.read_csv('data/for_training/train_set.csv')
    val = pd.read_csv('data/for_training/val_set.csv')
    test = pd.read_csv('data/for_training/test_set.csv')
    texts = pd.concat([train.text, val.text, test.text], axis=0)
    max_len = 0
    for text in texts:
        token_length = len(tokenizer.encode(text))
        if  token_length > max_len:
            max_len = token_length
    return max_len