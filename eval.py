from trainer import trainer 

from config import Config
config=Config("data/toutiao_cat_data.txt")
trainer=trainer(config)
trainer.predict("腾讯股价再创新高市值破3万亿，还有哪些“神股”只涨不跌？",model_path="models/model.pth")