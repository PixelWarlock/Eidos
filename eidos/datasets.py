import os
from torch.utils.data import Dataset
from eidos.transforms import *
from torchvision.transforms import Compose, ToTensor, Resize


class ImageDataset(Dataset):
    def __init__(self, src, transforms):
        self.transforms = transforms
        self.filepaths = [os.path.join(src, file) for file in os.listdir(src)]
        self.patchify = Patchify(patch_size=16)

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        sample = self.filepaths[index]
        x = self.transforms(sample)
        x = self.patchify(x)
        #show = Show()
        #show(x['patches'])
        return x
        
        