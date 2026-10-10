from transformers import BertModel, get_linear_schedule_with_warmup
import torch
from utils import Arguments
import torch.nn as nn

class Bert4TextClassification(nn.Module):
    def __init__(self,config:Arguments):
        super().__init__()
        self.bert_lr=config.bert_lr
        self.fc_lr=config.fc_lr
        self.weight_decay=config.weight_decay
        self.bert = BertModel.from_pretrained(config.model_dir)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.embedding_dim, config.class_num)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    def get_optimizer(self, num_training_steps, num_warmup_steps=0):
        optimizer = torch.optim.AdamW([
            {'params': self.bert.parameters(), 'lr': self.bert_lr},
            {'params': self.fc.parameters(), 'lr': self.fc_lr}
        ], weight_decay=self.weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, lr_scheduler
