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
        
        


        
    
    




