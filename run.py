from utils import load_data
from trainer import trainer 
from model import Bert4TextClassification
from ModelConfig import ModelConfig
if __name__ == "__main__":
    config=ModelConfig("config.json")
    model=Bert4TextClassification(config)
    optimizer, scheduler = model.get_optimizer()
    traindataLoader, devdataLoader, testdataLoader = load_data(config)
    trainer=trainer(config)
    trainer.train(traindataLoader, devdataLoader, testdataLoader, model, optimizer, scheduler)



