
label2id = {
    "100": 0,
    "101": 1,
    "102": 2,
    "103": 3,
    "104": 4,
    "106": 5,
    "107": 6,
    "108": 7,
    "109": 8,
    "110": 9,
    "112": 10,
    "113": 11,
    "114": 12,
    "115": 13,
    "116": 14
    
}

id2label = {
    0: "100",
    1: "101",
    2: "102",
    3: "103",
    4: "104",
    5: "106",
    6: "107",
    7: "108",
    8: "109",
    9: "110",
    10: "112",
    11: "113",
    12: "114",
    13: "115",
    14: "116"
}
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
import torch
data_dir="../bert-base-chinese"
tokenizer=BertTokenizer.from_pretrained(data_dir)
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
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
def load_data(file_path,batch_size=16):
    df = pd.read_csv(file_path, delimiter='_!_', header=None,engine="python")
    labels = df.iloc[:, 1]
   
    texts = df.iloc[:, 3]
    labels_clean = labels.astype(str).str.strip()
    labels_new = labels_clean.map(lambda x: int(label2id[x])).values
    encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=128, return_tensors="pt")
    labels_list = labels_new.tolist()
    encodings_list = [
        {key: value[i] for key, value in encodings.items()}
        for i in range(len(texts))
    ]
    train_data, test_data, train_labels, test_labels = train_test_split(encodings_list, labels_list, test_size=0.2, random_state=42)
    train_dataset = ToutiaoDataset(train_data, train_labels)
    test_dataset = ToutiaoDataset(test_data, test_labels)
    
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataLoader, test_dataLoader


if __name__ == "__main__":    
    train_dataLoader, test_dataLoader = load_data("toutiao_cat_data.txt")
    for batch in train_dataLoader:
        print(batch)
        break
        
