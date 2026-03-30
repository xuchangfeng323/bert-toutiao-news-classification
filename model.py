from transformers import BertModel
dir="../bert-base-chinese"

from torch import nn
class Bert(nn.Module):
    def __init__(self,embedding_dim,class_num,dropout_rate=0.1,model_dir="../bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, class_num)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
if __name__ == "__main__":
    model = Bert()
    print(model)