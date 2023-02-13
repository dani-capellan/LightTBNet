import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MCSZDataset(Dataset):
    """
    Montgomery (MC) & Shenzen (SZ) Dataset class
    Info taken from dict (.pkl)
    """

    def __init__(self, df, data, configs, do_transform=True, one_hot_encoding=True, apply_clahe=True):
        self.df = df
        self.data = data
        self.configs = configs
        self.do_transform = do_transform
        self.one_hot_encoding = one_hot_encoding
        self.apply_clahe = apply_clahe
        self.random_state = configs['random_state']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Get Image - ALL AP & without CLAHE (CLAHE applied in preprocess before CNN)
        key = "filename"
        img_path = self.df[key][idx]
        img = self.data[img_path]
        
        # Get Label
        label = self.df['TB_class'][idx]

        # Get highest value of img
        max_value = np.max(img)

        # Optional one hot encoding
        if self.one_hot_encoding:
            label = torch.nn.functional.one_hot(torch.from_numpy(np.array(label)),num_classes=self.configs['experimentEnv']['num_classes']) 
        else:
            label = torch.LongTensor([label])

        transform = A.Compose([
            A.CLAHE(clip_limit=(0.01*max_value,0.01*max_value), p=1) if self.apply_clahe else None,
            A.HorizontalFlip(p=0.5) if self.do_transform else None,  # Only during training
            A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5) if self.do_transform else None,
            A.Resize(height=self.configs['img_dim_ai'],width=self.configs['img_dim_ai']),  # Always applied
            A.Normalize(mean=0.5,std=0.5),  # Always applied
            ToTensorV2(),  # Always applied
        ])

        # Apply final transformation
        img_tr = transform(image=img)

        return img_tr['image'], label