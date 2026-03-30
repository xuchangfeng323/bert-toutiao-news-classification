from pyexpat import model
from utils import load_data
from transformers import get_scheduler
from model import Bert
import torch
from transformers import BertTokenizer
class Config:
    def __init__(self,data_path,num_epochs=1,batch_size=16,lr=1e-6,weight_decay=1e-2,device="cuda",model_dir="../bert-base-chinese",optimizer=None,loss_fn="cross_entropy"):
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_dataLoader, self.test_dataLoader = load_data(data_path,batch_size=batch_size)
        num_training_steps = self.num_epochs * len(self.train_dataLoader)
        self.model = Bert(embedding_dim=768,class_num=15,model_dir=model_dir)
        self.optimizer = optimizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
       
        self.scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
        )
        if loss_fn == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()

