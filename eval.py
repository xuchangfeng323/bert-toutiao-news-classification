from trainer import trainer 
from model import Bert
from ModelConfig import ModelConfig
if __name__ == "__main__":
    config=ModelConfig("config.json")
    trainer=trainer(config)
    model=Bert(config.embedding_dim, config.class_num, model_dir=config.model_dir)
    optimizer, scheduler = model.get_optimizer()
    
