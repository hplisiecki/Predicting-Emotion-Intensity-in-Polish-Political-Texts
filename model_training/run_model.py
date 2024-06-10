from model_training.dataset_and_model import Dataset, Model
from transformers import AutoTokenizer
from model_training.training_loop import training_loop
import wandb
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from model_training.dataset_wrangling import load_data, check_max_token_length

emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M', 'Irony_M']


df_train, df_val, df_test = load_data(emotion_columns)

hidden_dim = 768
dropout = 0.2
warmup_steps = 600
save_dir = 'models/test_run'

epochs = 1000
batch_size = 30
learning_rate = 5e-4
eps = 1e-8
weight_decay = 0.3
amsgrad = True
betas = (0.9, 0.999)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = 'deepsense-ai/trelbert'
tokenizer = AutoTokenizer.from_pretrained(model)

max_len = check_max_token_length(tokenizer)

model = Model(model_dir = model, metric_names = emotion_columns, dropout = dropout, hidden_dim = hidden_dim)

train, val, test = Dataset(tokenizer, df_train, max_len, emotion_columns), Dataset(tokenizer, df_val, max_len, emotion_columns), Dataset(tokenizer, df_test, max_len, emotion_columns)

train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)



# TRAINING SETTINGS
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, betas=betas)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=len(train_dataloader) * epochs)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

wandb.init(project="annotations_CNS",
           entity="hubertp")
wandb.watch(model, log_freq=5)

training_loop(model, optimizer, scheduler, epochs, train_dataloader, val_dataloader, criterion,
                device, save_dir, use_wandb=True)