from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd

class ToutiaoDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=128, use_keywords=False):
        if use_keywords:
            self.texts = [
    (str(data['title'][i]) if pd.notna(data['title'][i]) else '') + ' ' +
    (str(data['keywords'][i]) if pd.notna(data['keywords'][i]) else '')
    for i in range(len(data['title']))
]
        else:
            self.texts = data['title']
        self.labels = data['label_id']
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            'text': text,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    def collate_fn(self, batch):
        encodings = self.tokenizer([item['text'] for item in batch], 
                                   truncation=True, 
                                   padding='max_length', 
                                   max_length=self.max_length, 
                                   return_tensors="pt")
        return encodings['input_ids'], encodings['attention_mask'], torch.stack([item['labels'] for item in batch])
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

