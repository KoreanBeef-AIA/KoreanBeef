from models.customnet import CustomNet
from models.effnet import EffNet
from models.resnet import ResNet
from models.convnext import ConvNext
from models.inception import Inception
from models.deit import Deit
from models.effnet_v2_l import EffNet_l
from models.vit import VitBase

import timm

def get_model(model_name:str, model_args:dict):
    if model_name == 'Linear':
        return CustomNet(**model_args)
    if model_name == 'effnet':
        return EffNet(**model_args)
    if model_name == 'resnet':
        return ResNet(**model_args)
    if model_name == 'convnext':
        return ConvNext(**model_args)
    if model_name == 'inception':
        return Inception(**model_args)
    if model_name == 'deit':
        return Deit(**model_args)
    if model_name == 'effnet_v2':
        return EffNet_l(**model_args)
    if model_name == 'vitbase':
        return VitBase(**model_args)
    
if __name__ == '__main__':
    pass