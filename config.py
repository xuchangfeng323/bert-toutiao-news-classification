import json
import os
from utils import load_data
from transformers import get_scheduler
from model import Bert
import torch
from transformers import BertTokenizer


class ModelConfig:
    def __init__(self, config_path="config.json", data_path=None, num_epochs=None, 
                 batch_size=None, lr=None, weight_decay=None, device=None, 
                 model_dir=None, optimizer=None, loss_fn=None):
        
        default_config = self._load_json_config(config_path)
        self.num_epochs = num_epochs if num_epochs is not None else default_config.get("num_epochs", 1)
        self.batch_size = batch_size if batch_size is not None else default_config.get("batch_size", 16)
        self.lr = lr if lr is not None else default_config.get("lr", 1e-6)
        self.weight_decay = weight_decay if weight_decay is not None else default_config.get("weight_decay", 1e-2)
        device_str = device if device is not None else default_config.get("device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device_str
        self.model_dir = model_dir if model_dir is not None else default_config.get("model_dir", "../bert-base-chinese")
        self.loss_fn_name = loss_fn if loss_fn is not None else default_config.get("loss_fn", "cross_entropy") 
        self.train_dataLoader, self.test_dataLoader = load_data(self.data_path, batch_size=self.batch_size)
        num_training_steps = self.num_epochs * len(self.train_dataLoader)
        self.model = Bert(embedding_dim=768, class_num=15, model_dir=self.model_dir)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
 
        if self.loss_fn_name == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
    
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
            "loss_fn": self.loss_fn_name
        }