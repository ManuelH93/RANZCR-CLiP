# =============================================================
# Load modules
# =============================================================

import os
import time
import subprocess
import numpy as np
import pandas as pd
import ast
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchsummary import summary
from warmup_scheduler import GradualWarmupScheduler
import albumentations
import torch.cuda.amp as amp
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupKFold
import random

# =============================================================
# Set parameters
# =============================================================

CURRENTDIR = os.path.abspath('')
PARENTDIR = os.path.dirname(CURRENTDIR)
RAW_DATA = os.path.join(PARENTDIR,'raw_data')
OUTPUT = 'output'
PROCESSED = 'processed'
IMAGE_FOLDER = 'train'
SEED = 1

kernel_type = 'unet++b7_1024_lr1e4_bs4_30epo'
enet_type = 'timm-efficientnet-b7'
# If DEBUG == True, only run 1 epochs per fold
DEBUG = False
batch_size = 4 if not DEBUG else 1
init_lr = 1e-4
warmup_epo = 1
num_workers = 16 if not DEBUG else 0
cosine_epo = 29 if not DEBUG else 1
image_size = 1024 if not DEBUG else 224
n_epochs = warmup_epo + cosine_epo
use_amp = True if not DEBUG else False

log_dir = './logs'
model_dir = './models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log_{kernel_type}.txt')

scaler = amp.GradScaler(enabled=use_amp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}\n')

# =============================================================
# Set seed
# =============================================================

np.random.seed(SEED)
random.seed(SEED)

# =============================================================
# Load data
# =============================================================

df_train = pd.read_csv(os.path.join(PROCESSED,'train_v2.csv'))
df_train_anno = pd.read_csv(os.path.join(RAW_DATA,'train_annotations.csv'))

# =============================================================
# Reduce dataset if debug = True
# =============================================================

# If DEBUG == True, use only 5 samples with annotations to
# train model and one without annotations to test mask
# prediction.
df_train = pd.concat([
    df_train.query('w_anno == True and fold == 0').sample(1),
    df_train.query('w_anno == True and fold == 1').sample(1),
    df_train.query('w_anno == True and fold == 2').sample(1),
    df_train.query('w_anno == True and fold == 3').sample(1),
    df_train.query('w_anno == True and fold == 4').sample(1),
    df_train.query('w_anno == False and fold == 4').sample(1),
]) if DEBUG else df_train

print(f'\nShape of training dataframe: {df_train.shape}\n')

# =============================================================
# Define dataset
# =============================================================

class RANZCRDataset(Dataset):

    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(os.path.join(RAW_DATA, IMAGE_FOLDER, row.StudyInstanceUID + '.jpg'))[:, :, ::-1]

        if self.mode == 'test':
            mask = None
            res = self.transform(image=image)
        else:
            df_this = df_train_anno.query(f'StudyInstanceUID == "{row.StudyInstanceUID}"')
            mask = np.zeros((image.shape[0], image.shape[1], 2)).astype(np.uint8)
            for _, anno in df_this.iterrows():
                anno_this = np.array(ast.literal_eval(anno["data"]))
                mask1 = mask[:, :, 0].copy()
                mask1 = cv2.polylines(mask1, np.int32([anno_this]), isClosed=False, color=1, thickness=15, lineType=16)
                mask[:, :, 0] = mask1
                mask2 = mask[:, :, 1].copy()
                mask2 = cv2.circle(mask2, (anno_this[0][0], anno_this[0][1]), radius=15, color=1, thickness=25)
                mask2 = cv2.circle(mask2, (anno_this[-1][0], anno_this[-1][1]), radius=15, color=1, thickness=25)
                mask[:, :, 1] = mask2

            mask = cv2.resize(mask ,(image_size, image_size))
            mask = (mask > 0.5).astype(np.uint8)
            res = self.transform(image=image, mask=mask)

        image = res['image'].astype(np.float32).transpose(2, 0, 1) / 255.

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            mask = res['mask'].astype(np.float32)
            mask = mask.transpose(2, 0, 1).clip(0, 1)
            return torch.tensor(image), torch.tensor(mask)

# =============================================================
# Augment images
# =============================================================

transforms_train = albumentations.Compose([
    albumentations.Resize(image_size, image_size),                                    
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightness(limit=0.1, p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.75),
    albumentations.Cutout(max_h_size=int(image_size * 0.3), max_w_size=int(image_size * 0.3), num_holes=1, p=0.75),
])
transforms_val = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
])

# =============================================================
# Visualization
# =============================================================

df_show = df_train.query('w_anno==True').iloc[:8]
dataset_show = RANZCRDataset(df_show, 'train', transform=transforms_train)

f, axarr = plt.subplots(3, 3, figsize=(20,20))
masks = []
for p in range(3):
    img, mask = dataset_show[p]
    img[0] = img[0]
    axarr[0,p].imshow(img.transpose(0, 1).transpose(1,2))
    masks.append(mask)

for p in range(3):
    axarr[1,p].imshow(masks[p][0])

for p in range(3):
    axarr[2,p].imshow(masks[p][1])

plt.savefig(os.path.join(OUTPUT, 'annotations.png'), bbox_inches='tight')
#plt.show()
plt.clf()

# =============================================================
# Load model
# =============================================================

class SegModel(nn.Module):
    def __init__(self, backbone):
        super(SegModel, self).__init__()
        self.seg = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights='imagenet', classes=2, activation=None)
    def forward(self,x):
        global_features = self.seg.encoder(x)
        seg_features = self.seg.decoder(*global_features)
        seg_features = self.seg.segmentation_head(seg_features)
        return seg_features

m = SegModel(enet_type)
#print(f'\nModel output shape: {m(torch.rand(2,3,224,224)).shape}\n')

#print('\nModel summary\n')
#summary(m, input_size=(3, 224, 224), device="cpu")

# =============================================================
# Define loss
# =============================================================

criterion = nn.BCEWithLogitsLoss()

# =============================================================
# Learning rate scheduler
# =============================================================

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

optimizer = optim.Adam(m.parameters(), lr=init_lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
lrs = []
for epoch in range(1, n_epochs+1):
    scheduler_warmup.step(epoch-1)
    lrs.append(optimizer.param_groups[0]["lr"])
plt.figure(figsize=(20,3))
plt.plot(lrs)
plt.savefig(os.path.join(OUTPUT, 'lrs.png'), bbox_inches='tight')
#plt.show()
plt.clf()

# =============================================================
# Train and validation function
# =============================================================

def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    for (data, mask) in loader:

        optimizer.zero_grad()
        data, mask = data.to(device), mask.to(device)

        with amp.autocast(enabled=use_amp):
            logits = model(data)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_np = loss.item()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
        print(f'loss: {loss_np}, smth: {smooth_loss}')

    return np.mean(train_loss)


def valid_epoch(model, loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    with torch.no_grad():
        for (data, mask) in loader:
            data, mask = data.to(device), mask.to(device)
            logits = model(data)
            loss = criterion(logits, mask)
            val_loss.append(loss.item())
            LOGITS.append(logits.cpu())

    if get_output:
        LOGITS = torch.cat(LOGITS, 0).float().sigmoid()
        return LOGITS
    else:
        val_loss = np.mean(val_loss)
        return val_loss

# =============================================================
# Run
# =============================================================

def run(fold):
    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    train_ = df_train.query(f'w_anno==True and fold!={fold}').copy()
    valid_ = df_train.query(f'w_anno==True and fold=={fold}').copy()

    dataset_train = RANZCRDataset(train_, 'train', transform=transforms_train)
    dataset_valid = RANZCRDataset(valid_, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SegModel(enet_type)
    model = model.to(device)
    val_loss_min = np.Inf
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch-1)
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = valid_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if val_loss_min > val_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), model_file)
            val_loss_min = val_loss

for fold in range(5):
    run(fold)


# =============================================================
# Generate masks
# =============================================================

output_dir = f'mask_{kernel_type}'
os.makedirs(output_dir, exist_ok=True)

# Part 1, generate mask for those images with annotations. To
# prevent leaks, use only the model corresponding to the fold.
# =============================================================
for fold in range(5):
    valid_ = df_train.query(f'w_anno==True and fold=={fold}').copy()
    dataset_valid = RANZCRDataset(valid_, 'valid', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SegModel(enet_type)
    model = model.to(device)
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
    model.load_state_dict(torch.load(model_file), strict=False)
    model.eval()
    
    outputs = valid_epoch(model, valid_loader, get_output=True).numpy()

    for i, (_, row) in enumerate(valid_.iterrows()):
        png = (outputs[i] * 255).astype(np.uint8).transpose(1,2,0)
        # add a channel to make it able to be saved as .png
        png = np.concatenate([png, np.zeros((png.shape[0], png.shape[1], 1))], -1)
        cv2.imwrite(os.path.join(output_dir, f'{row.StudyInstanceUID}.png'), png)

# Part 2, For those images without annotations, use 5-fold
# models to predict and take the mean value.
# =============================================================
df_train_wo_anno = df_train.query(f'w_anno==False').copy().reset_index(drop=True)
dataset_test = RANZCRDataset(df_train_wo_anno, 'test', transform=transforms_val)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

models = []
for fold in range(5):
    model = SegModel(enet_type)
    model = model.to(device)
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
    model.load_state_dict(torch.load(model_file), strict=False)
    model.eval()
    models.append(model)

with torch.no_grad():
    for batch_id, data in enumerate(test_loader):
        data = data.to(device)
        outputs = torch.stack([model(data).sigmoid() for model in models], 0).mean(0).cpu().numpy()
        for i in range(outputs.shape[0]):
            row = df_train_wo_anno.loc[batch_id*batch_size + i]
            png = (outputs[i] * 255).astype(np.uint8).transpose(1,2,0)
            # add a channel to make it able to be saved as .png
            png = np.concatenate([png, np.zeros((png.shape[0], png.shape[1], 1))], -1)
            cv2.imwrite(os.path.join(output_dir, f'{row.StudyInstanceUID}.png'), png)
