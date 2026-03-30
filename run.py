
from trainer import trainer 

from config import Config
config=Config("data/toutiao_cat_data.txt")
trainer=trainer(config)
trainer.train()



