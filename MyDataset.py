from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

class ToutiaoDataset(Dataset):
    def __init__(self, texts, labels,batch_size=16, tokenizer=BertTokenizer.from_pretrained('../bert-base-chinese'), max_length=128):
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            'text': text,
            'labels': torch.tensor(label)
        }
    def collate_fn(self, batch):
        encodings = self.tokenizer([item['text'] for item in batch], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.stack([item['labels'] for item in batch])
        }
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)
