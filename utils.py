import os
from swanlab import config
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
import torch
from MyDataset import ToutiaoDataset
from arguments import Arguments
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import json


def load_data(config):
    data_dir=config.data_path
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_dev = pd.read_csv(os.path.join(data_dir, 'dev.csv'))
    train_data = df_train['text'].tolist()
    train_labels = df_train['label'].tolist()
    test_data = df_test['text'].tolist()
    test_labels = df_test['label'].tolist()
    dev_data = df_dev['text'].tolist()
    dev_labels = df_dev['label'].tolist()
    tokenizer = BertTokenizer.from_pretrained(config.model_dir)
    train_dataset = ToutiaoDataset(train_data, train_labels,tokenizer,config.max_length)
    test_dataset = ToutiaoDataset(test_data, test_labels,tokenizer,config.max_length)
    dev_dataset = ToutiaoDataset(dev_data, dev_labels,tokenizer,config.max_length)

    train_dataLoader = train_dataset.get_data_loader(batch_size=config.batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    test_dataLoader = test_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    return train_dataLoader, dev_dataLoader, test_dataLoader


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
class EarlyStop():
    def __init__(self,config):
        self.config=config
        self.monitor = config.monitor
        self.delta=config.delta
        self.best_score = None
        self.counter = 0
        self.patience = config.patience
        self.early_stop = False
    def __call__(self, epoch,loss,acc, model,optimizer,scheduler,):
        if self.monitor == 'val_acc':
            if self.best_score is None :
                self.best_score = acc
                
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc)
            

            if acc-self.best_score  < self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = acc
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc)
        elif self.monitor == 'val_loss':
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss)
            if self.best_score - loss > self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = loss
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss)
        return self.early_stop
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, dev_metrics):
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = os.path.join(self.config.model_save_path, checkpoint_name)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"保存 epoch {epoch + 1} 的 checkpoint: {checkpoint_path}")
        self.best_model_path = checkpoint_path
        print(f"更新最佳模型: {checkpoint_path} (监控指标 '{self.monitor}' = {dev_metrics:.6f})")
    
        

if __name__ == '__main__': 
    args = Arguments("arguments.json")
    train_dataLoader, dev_dataLoader, test_dataLoader = load_data(args)
    for batch in train_dataLoader:
        print(batch)
        break
    
        
        


        
    
    






