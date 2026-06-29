from __future__ import annotations
import os
from PIL import Image
from torch.utils.data import Dataset
from eidos.transforms.base import BaseTransform
from eidos.transforms.mask import MaskToGram


class EidosDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 mask_dir, 
                 shared_transforms=None,
                 patch_size=16):
        
        self.data_dict = self._data_dict(img_dir=img_dir, mask_dir=mask_dir)
        self.shared_transforms = shared_transforms
        
        # default
        self.base_transforms = BaseTransform()
        self.mask_transform = MaskToGram(patch_size=patch_size)

    def _data_dict(self, img_dir, mask_dir):
        img_filepaths = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir)])
        mask_filepaths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)])
        self.data_dict = dict(zip(img_filepaths, mask_filepaths))

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        img_filepath, mask_filepath = self.data_dict.items()[index]
        img = Image.open(img_filepath).convert('RBG')
        mask = Image.open(mask_filepath)

        # shared
        if self.shared_transforms:
            img, mask = self.shared_transforms(img,mask)
        
        # base
        img = self.base_transforms(img)

        # mask
        gram = self.mask_transform(mask)
        
        return img, gram
