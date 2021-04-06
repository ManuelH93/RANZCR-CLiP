import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import cv2
import time
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import timm

import helper


###########################################################
# Define parameters
###########################################################

CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
PARENTDIR = os.path.dirname(CURRENTDIR)
RAW_DATA = os.path.join(PARENTDIR,'raw_data')
TRAIN = 'train'
OUTPUT = 'output'
SEED = 'some'
train = pd.read_csv(os.path.join(RAW_DATA,'train.csv'))
train = train.head(3)
val = train.sample(frac=0.3)
target_cols=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']

###########################################################
# Set seed
###########################################################

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_some(seed):
    random.seed(seed)
    torch.manual_seed(seed)

if SEED == 'all':
    print("[ Seed setting : slow and reproducible ]")
    seed_all(2001)
else:
    print("[ Seed setting : fast and random ]")
    seed_some(2001)

###########################################################
# Exploratory Data Analysis
###########################################################

def EDA(train):
    
    print(train.head())
    print(train.tail())
    print("Number of images: ", train["StudyInstanceUID"].shape[0])
    print("Number of unique images: ", train["StudyInstanceUID"].unique().shape[0])
    print("Number of unique patients: ", train["PatientID"].unique().shape[0])

    # Distribution of labels
    plt.figure(figsize=(8, 8))
    df_tmp = train.iloc[:, 1:-1].sum()
    sns.barplot(x=df_tmp.values, y=df_tmp.index)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Number of images", fontsize=15)
    plt.title("Distribution of labels", fontsize=16)
    plt.savefig(os.path.join(OUTPUT, 'dist_labels.png'), bbox_inches='tight')
    #plt.show()
    plt.clf

    # Distribution of observations by patient ID
    plt.figure(figsize=(16, 6))
    df_tmp = train["PatientID"].value_counts()
    sns.countplot(x=df_tmp.values)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=14)
    plt.xlabel("Number of observations", fontsize=15)
    plt.ylabel("Number of patients", fontsize=15)
    plt.title("Distribution of observations by PatientID", fontsize=16)
    plt.savefig(os.path.join(OUTPUT, 'dist_obs.png'), bbox_inches='tight')
    #plt.show()

#EDA(train)

###########################################################
# Define dataset
###########################################################

class RANZCR_Dataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.labels = df[target_cols].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{os.path.join(RAW_DATA, TRAIN, file_name)}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).float()
        return image, label

###########################################################
# Test if dataset load works
###########################################################

#ds = RANZCR_Dataset(train, transform=helper.get_transforms(data='train'))

#for i in range(5):
#    image, label = ds[i]
#    image = image.numpy()
#    print(image.shape)
#    print(image.min(), image.max(), image.mean(), image.std())
#    plt.imshow(image[0], cmap='gray')
#    plt.title(f'label: {label}')
#    plt.show()

###########################################################
# Load test and validation dataset
###########################################################

train_set = RANZCR_Dataset(train, transform=helper.get_transforms(data='train'))
val_set = RANZCR_Dataset(val, transform=helper.get_transforms(data='valid'))

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 5

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)

###########################################################
# Load model
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomResNext(pretrained=True)
model = model.to(device)

summary(model, input_size=(3, 640, 640))

###########################################################
# Define loss calculation
###########################################################

def calc_loss(pred, target, metrics):
    bce = F.binary_cross_entropy_with_logits(pred, target) 
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)

    return bce

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))  

###########################################################
# Define training
###########################################################

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    early_stopping = False

    # for figure
    epochs = []
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        epochs.append(epoch+1)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            iter = 0
            for inputs, labels in dataloaders[phase]:
                iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    if iter%250 == 0:
                        print("Epoch:", epoch+1, "- Phase:", phase, "- iteration:", iter)
                        print(loss)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
          
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['bce'] / epoch_samples
           
            # collect statistics for figure and take lr step
            if phase == 'train':
                print(f'saving epoch {epoch} model')
                torch.save(model.state_dict(), os.path.join(OUTPUT, f'model_epoch_{epoch}.pth'))
                train_loss.append(metrics['bce']/epoch_samples)
                scheduler.step()
            else:
                val_loss.append(metrics['bce']/epoch_samples)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            elif phase == 'val' and epoch_loss >= best_loss:
                epochs_no_improve += 1
                if epochs_no_improve == 500:
                    print('Early stopping!')
                    early_stopping = True

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if early_stopping == True:
            break
        else:
            continue
    print('Best val loss: {:4f}'.format(best_loss))

    # Save loss figure
    plt.plot(epochs, train_loss, color='g', label = 'train')
    plt.plot(epochs, val_loss, color='orange', label = 'test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(OUTPUT, 'losses.png'))
    #plt.show()
    plt.clf()

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(OUTPUT, 'best_model.pth'))
    return model

###########################################################
# Run model
###########################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = CustomResNext(pretrained=True).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=1)

###########################################################
# Predict
###########################################################

model.eval()   # Set model to evaluate mode

train = train.head(2)
test_dataset = RANZCR_Dataset(train, transform=helper.get_transforms(data='test'))

# Important to keep batch size equalt to one, as each image gets
# split into several tiles and is then put back together
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

for inputs in test_loader:
    inputs = inputs[0].to(device)
    preds = model(inputs)
    print(preds)
    # Create class probabilities
    preds = preds.sigmoid().detach().numpy()
    print(preds)