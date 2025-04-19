import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AISD_dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".tif"))])
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.imgs[index]))
        mask = Image.open(os.path.join(self.mask_dir, self.imgs[index]))

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, torch.tensor(np.array(mask))