import torch
from torch import nn
import numpy as np
from transformers import RobertaModel, AutoModel, BertModel

###############################################################################
"""
Dataset and model classes.
"""
###############################################################################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, df, max_len, metric_names):
        """
        Initialize the dataset.
        :param tokenizer:  the tokenizer to use
        :param df: the dataframe to use
        :param max_len:  the maximum length of the sequences
        :param metric_names: the names of the metrics to use
        """
        self.tokenizer = tokenizer
        self.metric_names = metric_names
        for name in self.metric_names:
            setattr(self, 'labels_' + name, eval("df['norm_" + name + "'].values.astype(float)"))

        # check if tokenizer is a list
        self.texts = [self.tokenizer(str(text),
                                     padding='max_length', max_length=max_len, truncation=True,
                                     return_tensors="pt") for text in df['text']]

    def classes(self):
        """
        Return the classes of the dataset.
        :return: the classes
        """
        exec_string = ', '.join(["self.labels_" + name for name in self.metric_names])
        return eval(exec_string)

    def __len__(self):
        """
        Return the length of the dataset.
        :return: the length
        """
        setattr(self, 'length', eval("len(self.labels_" + self.metric_names[0] + ")"))
        return self.length

    def get_batch_labels(self, idx):
        """
        Return the labels of the batch.
        :param idx: the index of the batch
        :return: the labels
        """
        exec_string = ', '.join(["np.array(self.labels_" + name + "[idx])" for name in self.metric_names])
        return eval(exec_string)

    def get_batch_texts(self, idx):
        """
        Return the texts of the batch.
        :param idx: the index of the batch
        :return: the texts
        """
        return self.texts[idx]

    def __getitem__(self, idx):
        """
        Return the item at the given index.
        :param idx: the index
        :return: the item
        """
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class Model(torch.nn.Module):

    def __init__(self, model_dir, metric_names, dropout=0.2, hidden_dim=768):
        """
        Initialize the model.
        :param model_name: the name of the model
        :param metric_names: the names of the metrics to use
        :param dropout: the dropout rate
        :param hidden_dim: the hidden dimension of the model
        """
        super(Model, self).__init__()

        self.metric_names = metric_names
        self.bert = AutoModel.from_pretrained(model_dir)


        for name in self.metric_names:
            setattr(self, name, nn.Linear(hidden_dim, 1))
            setattr(self, 'l_1_' + name, nn.Linear(hidden_dim, hidden_dim))

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):
        """
        Forward pass of the model.
        :param args: the inputs
        :return: the outputs
        """
        _, x = self.bert(input_ids = input_id, attention_mask=mask, return_dict=False)
        output = self.rate_embedding(x)
        return output

    def rate_embedding(self, x):
        output_ratings = []
        for name in self.metric_names:
            first_layer = self.relu(self.dropout(self.layer_norm(getattr(self, 'l_1_' + name)(x) + x)))
            second_layer = self.sigmoid(getattr(self, name)(first_layer))
            output_ratings.append(second_layer)

        return output_ratings
