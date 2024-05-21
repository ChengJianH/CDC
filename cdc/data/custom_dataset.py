"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            try:
                self.val_transform = transform['val']
            except:
                pass
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)

        sample = {}
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)
        try:
            sample['val'] = self.val_transform(image)
        except:
            pass
        sample['index']= index
        sample['target']= label

        return sample