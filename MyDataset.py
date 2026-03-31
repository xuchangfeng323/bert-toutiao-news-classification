from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
import torch
class ToutiaoDataset(Dataset):
    def __init__(self, encodings, labels):
        
        self.encodings = encodings
        self.labels = labels
        
        

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        label = self.labels[idx]
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label)
        }