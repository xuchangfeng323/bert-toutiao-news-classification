
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
from MyDataset import ToutiaoDataset

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
def load_data(file_path,batch_size=16):
    df = pd.read_csv(file_path, delimiter='_!_', header=None,engine="python")
    labels = df.iloc[:, 1]
   
    texts = df.iloc[:, 3]
    labels_clean = labels.astype(str).str.strip()
    labels_new = labels_clean.map(lambda x: int(label2id[x])).values
   
    labels_list = labels_new.tolist()
    
    train_data, testanddev_data, train_labels, testanddev_labels = train_test_split(texts.tolist(), labels_list, test_size=0.3, random_state=42, stratify=labels_list)
    test_data, dev_data, test_labels, dev_labels = train_test_split(testanddev_data, testanddev_labels, test_size=0.5, random_state=42, stratify=testanddev_labels)
    train_dataset = ToutiaoDataset(train_data, train_labels)
    test_dataset = ToutiaoDataset(test_data, test_labels)
    dev_dataset = ToutiaoDataset(dev_data, dev_labels)
    train_dataLoader = train_dataset.get_data_loader(batch_size=batch_size)
    test_dataLoader = test_dataset.get_data_loader(batch_size=batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=batch_size)
    return train_dataLoader, test_dataLoader, dev_dataLoader


if __name__ == "__main__":    
    train_dataLoader, test_dataLoader = load_data("./data/toutiao_cat_data.txt")
    for batch in train_dataLoader:
        print(batch)
        break
        
