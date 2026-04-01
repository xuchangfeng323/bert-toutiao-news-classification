
from trainer import trainer 

from config import ModelConfig
if __name__ == "__main__":
    config=ModelConfig("config.json")
    trainer=trainer(config)
    trainer.train()



