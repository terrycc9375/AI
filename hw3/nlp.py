# Neural Network Library for NLP
import torch

# Pretrained models and tokenizers
import transformers

# Data processers
import pandas as pd
import json
import argparse

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, tokenizer: transformers.AutoTokenizer, max_length: int):
        file = pd.read_csv(csv_path)
        self.texts = file['text'].tolist()
        self.labels = file['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       text = self.texts[idx]
       label = self.labels[idx]
       encoded = self.tokenizer(
           text,
           padding='max_length',
           truncation=True,
           max_length=self.max_length,
           return_tensors='pt'
       )
       item = {
           'input_ids': encoded['input_ids'].squeeze(0),
           'attention_mask': encoded['attention_mask'].squeeze(0),
           'label': torch.tensor(label, dtype=torch.long)
       }
       return item
