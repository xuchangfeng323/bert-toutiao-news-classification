import json
import os
import torch
from transformers import BertTokenizer
class Arguments:
    def __init__(self, config_path="arguments.json"):
        args_dict = self._load_json_config(config_path)
        for key, value in args_dict.items():
            setattr(self, key, value)
        
    def _load_json_config(self, config_path):
       
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
