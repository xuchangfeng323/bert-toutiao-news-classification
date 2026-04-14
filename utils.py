
from swanlab import config
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
from ModelConfig import ModelConfig

from sklearn.model_selection import train_test_split

def load_data(config):
    batch_size=config.batch_size
    df = pd.read_csv(config.data_dir, delimiter='_!_', header=None,engine="python")
    labels = df.iloc[:, 1]
   
    texts = df.iloc[:, 3]
    labels_clean = labels.astype(str).str.strip()
    labels_new = labels_clean.map(lambda x: int(label2id[x])).values
   
    labels_list = labels_new.tolist()
    
    train_data, testanddev_data, train_labels, testanddev_labels = train_test_split(texts.tolist(), labels_list, test_size=0.3, random_state=42, stratify=labels_list)
    test_data, dev_data, test_labels, dev_labels = train_test_split(testanddev_data, testanddev_labels, test_size=0.5, random_state=42, stratify=testanddev_labels)
    train_dataset = ToutiaoDataset(train_data, train_labels,config.max_length)
    test_dataset = ToutiaoDataset(test_data, test_labels,config.max_length)
    dev_dataset = ToutiaoDataset(dev_data, dev_labels,config.max_length)
    train_dataLoader = train_dataset.get_data_loader(batch_size=config.batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=config.batch_size)
    test_dataLoader = test_dataset.get_data_loader(batch_size=config.batch_size)
    return train_dataLoader, dev_dataLoader, test_dataLoader    

import pandas as pd
import torch
class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] 
                                  for _ in range(self.num_classes)]
        
    def add(self, predictions, labels):    
        predictions = predictions.tolist()
        labels = labels.tolist()

        for pred, true in zip(predictions, labels):
            self.confusion_matrix[true][pred] += 1
    def reset(self):
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] 
                                  for _ in range(self.num_classes)]
    
    def calculate_tp_fp_fn(self, class_id): 
        tp = self.confusion_matrix[class_id][class_id]
        fp = sum(self.confusion_matrix[i][class_id] for i in range(self.num_classes)) - tp
        fn = sum(self.confusion_matrix[class_id]) - tp
        return tp, fp, fn
    
    def precision(self, class_id=None):
        if class_id is not None:
            tp, fp, _ = self.calculate_tp_fp_fn(class_id)
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)
        else :
            total_tp,total_fp,total_fn = 0,0,0
            for i in range(self.num_classes):
                tp, fp, fn = self.calculate_tp_fp_fn(i)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if total_tp + total_fp == 0:
                return 0.0
            return total_tp /(total_tp + total_fp)
        

    
    def recall(self, class_id=None):
       
        if class_id is not None:
            tp, fp, fn = self.calculate_tp_fp_fn(class_id)
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)
        else:
            total_tp,total_fp,total_fn = 0,0,0
            for i in range(self.num_classes):
                tp, fp, fn = self.calculate_tp_fp_fn(i)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if total_tp + total_fn == 0:
                return 0.0
            return total_tp /(total_tp + total_fn)
    
    def f1_score(self, class_id=None):
       
        if class_id is not None:
            p = self.precision(class_id)
            r = self.recall(class_id)
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)
        
        else:
            total_tp,total_fp,total_fn = 0,0,0
            for i in range(self.num_classes):
                tp, fp, fn = self.calculate_tp_fp_fn(i)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if total_tp + total_fn == 0:
                r = 0.0
            else:
                r = total_tp /(total_tp + total_fn)
            if total_tp + total_fp == 0:
                p = 0.0
            else:
                p = total_tp /(total_tp + total_fp)
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)
    def get_results(self):
        p_list=[]
        r_list=[]
        f1_list=[]
        for i in range(self.num_classes):
            p_list.append(self.precision(i))
            r_list.append(self.recall(i))
            f1_list.append(self.f1_score(i))
        df=pd.DataFrame({'precision':p_list,'recall':r_list,'f1_score':f1_list})
        df.loc['macro_avg'] = df.mean()
        p=self.precision()
        r=self.recall()
        f1=self.f1_score()
        df.loc['micro_avg'] = [p,r,f1]
        return df

if __name__ == '__main__': 
    metrics1 = Metrics(num_classes=3)
    labels1 = torch.tensor([0, 0, 1, 1, 2, 2])
    preds1  = torch.tensor([0, 1, 1, 1, 2, 0]) 
    metrics1.add(preds1, labels1)
    results1 = metrics1.get_results()
    print(results1)
        
        


        
    
    





if __name__ == "__main__":    
    from ModelConfig import ModelConfig
    config = ModelConfig()
    train_dataLoader, dev_dataLoader, test_dataLoader = load_data(config)
    for batch in train_dataLoader:
        print(batch)
        break
        
