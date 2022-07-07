import os
import numpy as np
import torch
from PIL import Image

class SiameseLongTailedDataset(torch.utils.data.Dataset):
    """Siamese long-tailed dataset generated from the txt file"""
    """Two views for training"""
    def __init__(self, img_root_path, txt_path, num_classes, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt_path) as f:
            for line in f:
                self.img_path.append(os.path.join(img_root_path, line.split(' ')[0]))
                self.targets.append(int(line.split(' ')[1]))
        
        self.num_classes = num_classes
        self.cls_num_list = [0] * self.num_classes
        for i in range(len(self.targets)):
            self.cls_num_list[self.targets[i]] += 1
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            img_org = self.transform[0](sample)
            img_cont = self.transform[1](sample)
        return img_org, img_cont, target
    

class LongTailedDataset(torch.utils.data.Dataset):
    """Long-tailed dataset generated from the txt file"""
    """Single view for evaluation"""
    def __init__(self, img_root_path, txt_path, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt_path) as f:
            for line in f:
                self.img_path.append(os.path.join(img_root_path, line.split(' ')[0]))
                self.targets.append(int(line.split(' ')[1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            img_org = self.transform(sample)
        return img_org, target
