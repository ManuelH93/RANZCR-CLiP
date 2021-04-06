import albumentations as A
from albumentations.pytorch import ToTensorV2

###########################################################
# Transforms
###########################################################

def get_transforms(data, p=1.0):
    if data == 'train':
        return A.Compose([
            #Resize(600, 600),
            A.RandomResizedCrop(640, 640, scale=(0.85, 1.0), p=1.0),
            A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                # These are the mean and std values of the Imagenet
                # dataset. We use those values as we use a model that
                # was pre-trained on the Imagenet data. 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return A.Compose([
            A.RandomResizedCrop(640, 640, scale=(0.85, 1.0), p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'test':
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])