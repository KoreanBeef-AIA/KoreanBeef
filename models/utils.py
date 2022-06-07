from models.customnet import CustomNet
from models.effnet import EffNet
import timm

def get_model(model_name:str, model_args:dict):
    if model_name == 'Linear':
        return CustomNet(**model_args)
    if model_name == 'effnet':
        return EffNet(**model_args)
    
if __name__ == '__main__':
    pass