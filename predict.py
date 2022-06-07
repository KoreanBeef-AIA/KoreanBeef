"""Predict
"""
from modules.utils import load_yaml, save_pickle, save_json
from modules.datasets import TestDataset
from models.utils import get_model

from torch.utils.data import DataLoader

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import pandas as pd

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'predict_config.yaml'))


# Serial
train_serial = predict_config['TRAIN']['train_serial']
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + '_' + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

# Data Directory
DATA_DIR = os.path.join(predict_config['DIRECTORY']['dataset'],predict_config['DIRECTORY']['phase'])
SAMPLE_DIR = predict_config['DIRECTORY']['sample']

# Recorder Directory
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

# Train config
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

# Gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])

if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    test_dataset = TestDataset(img_folder=os.path.join(DATA_DIR, 'images'),
                                dfpath=os.path.join(DATA_DIR, 'test_images.csv'))
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=train_config['DATALOADER']['batch_size'],
                                num_workers=train_config['DATALOADER']['num_workers'], 
                                shuffle=False,
                                pin_memory=train_config['DATALOADER']['pin_memory'],
                                drop_last=train_config['DATALOADER']['drop_last'])

    # Load model
    model_name = train_config['TRAINER']['model']
    model_args = train_config['MODEL'][model_name]
    model = get_model(model_name=model_name, model_args=model_args).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    # Make predictions
    y_preds = []
    filenames = []

    for batch_index, (x, filename) in enumerate(tqdm(test_dataloader)):
        x = x.to(device, dtype=torch.float)
        y_logits = model(x).cpu()
        y_pred = torch.argmax(y_logits, dim=1)
        y_logits = y_logits.detach().numpy()
        y_pred = y_pred.detach().numpy()
        for fname in filename:
            filenames.append(fname)
        for yp in y_pred:
            y_preds.append(yp)
    
    # Decode Prediction Labels
    label_decoding = {0:'1++', 1:'1+', 2:'1', 3:'2', 4:'3'}
    pred_df = pd.DataFrame(list(zip(filenames, y_preds)), columns=['id','grade'])
    pred_df['grade'] = pred_df['grade'].replace(label_decoding)
    
    # Reorder 
    sample_df = pd.read_csv(SAMPLE_DIR)
    sorter = list(sample_df['id'])
    resdf = pred_df.set_index('id')
    result = resdf.loc[sorter].reset_index()
    
    # Save predictions
    resultpath = os.path.join(PREDICT_DIR, 'predictions.csv')
    result.to_csv(resultpath, index=False)
    print('Done')