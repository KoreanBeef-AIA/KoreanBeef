from itertools import product
import logging
import random
import pickle
import shutil
import json
import yaml
import csv
import os

'''
File IO
'''

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_json(path, obj, sort_keys=True)-> str:  
    try:
        with open(path, 'w') as f:    
            json.dump(obj, f, indent=4, sort_keys=sort_keys)
        msg = f"Json saved {path}"
    except Exception as e:
        msg = f"Fail to save {e}"
    return msg

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, sort_keys=False)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

'''
Logger
'''
def get_logger(name: str, dir_: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(dir_, f'{name}.log'))
    
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

if __name__ == '__main__':
    pass