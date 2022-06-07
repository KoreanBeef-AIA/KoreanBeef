import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from modules.utils import load_yaml, save_yaml
import zipfile


# Project directory
PROJECT_DIR = os.path.dirname(__file__)
PREPROCESS_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/preprocess_config.yaml')

# Preprocess config
config = load_yaml(PREPROCESS_CONFIG_PATH)

# Data directory
DATA_DIR = config['DIRECTORY']['dataset']

# Make directories
SRC_DIR = os.path.join(DATA_DIR, '00_source')
SPLIT_DIR = os.path.join(DATA_DIR, '01_splitdataset')
SPLIT_TRAIN_DIR = os.path.join(SPLIT_DIR, 'train')
SPLIT_VALID_DIR = os.path.join(SPLIT_DIR, 'val')
SPLIT_TEST_DIR = os.path.join(SPLIT_DIR, 'test')

os.mkdir(SRC_DIR)
os.mkdir(SPLIT_DIR)
os.mkdir(SPLIT_TRAIN_DIR)
os.mkdir(SPLIT_VALID_DIR)
os.mkdir(SPLIT_TEST_DIR)


# Unzip datafiles
trainpath = os.path.join(DATA_DIR, 'train.zip')
testpath = os.path.join(DATA_DIR, 'test.zip')

with zipfile.ZipFile(trainpath, 'r') as zip_ref:
    zip_ref.extractall(SRC_DIR)

with zipfile.ZipFile(testpath, 'r') as zip_ref:
    zip_ref.extractall(SPLIT_TEST_DIR)



# Split training data into train/val
LABEL_PATH = os.path.join(SRC_DIR, 'grade_labels.csv')
data = pd.read_csv(LABEL_PATH)

X = data['imname']
y = data['grade']

X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size = 0.2, shuffle=True, random_state=42)

train = pd.DataFrame(list(zip(X_train, y_train)),columns=['imname','grade'])
valid = pd.DataFrame(list(zip(X_valid, y_valid)),columns=['imname','grade'])


# Save train and validation labels
train.to_csv(os.path.join(SPLIT_TRAIN_DIR, 'grade_labels.csv'), index=False)
valid.to_csv(os.path.join(SPLIT_VALID_DIR, 'grade_labels.csv'), index=False)

# Copy images
## Train
os.mkdir(os.path.join(SPLIT_TRAIN_DIR,'images'))
for imname in tqdm(X_train):
    src = os.path.join(SRC_DIR,'images',imname)
    dst = os.path.join(SPLIT_TRAIN_DIR,'images',imname)
    shutil.copy(src, dst)
    
## Valid
os.mkdir(os.path.join(SPLIT_VALID_DIR,'images'))
for imname in tqdm(X_valid):
    src = os.path.join(SRC_DIR,'images',imname)
    dst = os.path.join(SPLIT_VALID_DIR, 'images', imname)
    shutil.copy(src,dst)


'''
SRC_DIR = os.path.join(PROJECT_DIR, 'data', '00_source')
DST_DIR = os.path.join(PROJECT_DIR, 'data', '01_splitdataset')
SRC_IMG_DIR = os.path.join(SRC_DIR, 'images')

# Read in label dataframe
LABEL_PATH = os.path.join(SRC_DIR, 'grade_labels.csv')
data = pd.read_csv(LABEL_PATH)

# Split dataset
X = data['imname']
y = data['grade']

X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size = 0.2, shuffle=True, random_state=42)

train = pd.DataFrame(list(zip(X_train, y_train)),columns=['imname','grade'])
valid = pd.DataFrame(list(zip(X_valid, y_valid)),columns=['imname','grade'])

# Save train and validation labels
train.to_csv(os.path.join(DST_DIR, 'train', 'grade_labels.csv'), index=False)
valid.to_csv(os.path.join(DST_DIR, 'val', 'grade_labels.csv'), index=False)

# Copy images
## Train
for imname in tqdm(X_train):
    src = os.path.join(SRC_IMG_DIR, imname)
    dst = os.path.join(DST_DIR, 'train', 'images', imname)
    shutil.copy(src, dst)

for imname in tqdm(X_valid):
    src = os.path.join(SRC_IMG_DIR, imname)
    dst = os.path.join(DST_DIR, 'val', 'images', imname)
    shutil.copy(src,dst)
'''