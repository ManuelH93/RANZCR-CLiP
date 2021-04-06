###########################################################
# Import libraries
###########################################################

import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import copy
import timm

import helper

###########################################################
# Set parameters
###########################################################

MODEL = 'trained_model'
MODEL_VERSION = '2021.03.13_base_line'

RAW_DATA = 'raw_data'
TEST = 'test'
OUTPUT = 'output'
test = pd.read_csv(os.path.join(RAW_DATA,'sample_submission.csv'))
target_cols=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###########################################################
# Define dataset
###########################################################

class RANZCR_Dataset_test(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{os.path.join(RAW_DATA, TEST, file_name)}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

###########################################################
# Define model
###########################################################

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnet200d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, len(target_cols))

    def forward(self, x):
        x = self.model(x)
        return x

###########################################################
# Load model
###########################################################

model = CustomResNext(pretrained=False).to(device)

model.load_state_dict(torch.load(os.path.join(MODEL,MODEL_VERSION,'best.model'), map_location=torch.device(device)))

model.eval()   # Set model to evaluate mode

###########################################################
# Make predictions
###########################################################

test = test.head(2)
test_dataset = RANZCR_Dataset_test(test, transform=helper.get_transforms(data='test'))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

predictions = []
for inputs in test_loader:
    inputs = inputs.to(device)
    preds = model(inputs)
    preds = preds.sigmoid().detach().numpy()
    predictions.append(preds)
predictions = np.concatenate(predictions)

target_cols = test.iloc[:, 1:12].columns.tolist()
test[target_cols] = predictions
test[['StudyInstanceUID'] + target_cols].to_csv(os.path.join(OUTPUT,'submission.csv'), index=False)