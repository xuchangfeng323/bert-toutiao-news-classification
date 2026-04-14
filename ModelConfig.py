import json
import os
from transformers import get_scheduler
import torch
from transformers import BertTokenizer
class ModelConfig:
    def __init__(self, config_path="config.json", data_path=None, num_epochs=None, 
                 batch_size=None, lr=None, weight_decay=None, device=None, 
                 model_dir=None, optimizer=None, loss_fn=None):
        
        default_config = self._load_json_config(config_path)
        self.num_epochs = num_epochs if num_epochs is not None else default_config.get("num_epochs", 1)
        self.batch_size = batch_size if batch_size is not None else default_config.get("batch_size", 16)
        self.learning_rate = lr if lr is not None else default_config.get("learning_rate", 1e-6)
        self.weight_decay = weight_decay if weight_decay is not None else default_config.get("weight_decay", 1e-2)
        device_str = device if device is not None else default_config.get("device", "cuda")
        self.embedding_dim = default_config.get("embedding_dim", 768)
        self.class_num = default_config.get("class_num", 15)
        self.max_length = default_config.get("max_length", 128)
        self.dropout_rate = default_config.get("dropout_rate", 0.1)
        self.num_workers = default_config.get("num_workers", 4)
        if device_str == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device_str
        self.loss_fn=None
        self.loss_fn_name = loss_fn if loss_fn is not None else default_config.get("loss_fn", "cross_entropy") 
        self.data_dir = default_config.get("data_dir", "data/toutiao_cat_data.txt")
        self.model_dir = default_config.get("model_dir", "../bert-base-chinese")
        if self.loss_fn_name == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.patience = default_config.get("patience", 5)
        self.monitor = default_config.get("monitor", "val_acc")
        self.delta = default_config.get("delta", 0.01)
        self.model_save_path = default_config.get("model_save_path", "./checkpoints")
        self.label_mapping_path=default_config.get("label_mapping_path", {})
        
    def _load_json_config(self, config_path):
       
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def to_dict(self):
        return {
            "data_path": self.data_path,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "model_dir": self.model_dir,
            "loss_fn": self.loss_fn_name,
            "patience": self.patience,
            "monitor": self.monitor,
            "delta": self.delta,
            "model_save_path": self.model_save_path,
            "label_mapping_path": self.label_mapping_path,

        }