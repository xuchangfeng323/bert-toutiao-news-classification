
from trainer import trainer 

from config import Config
if __name__ == "__main__":
    config=Config("data/toutiao_cat_data.txt")
    trainer=trainer(config)
    trainer.train()



