from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from eidos.transforms import *
from eidos.datasets import ImageDataset


def main():
    
    transforms = Compose([
        LoadImage(),
        Resize((256,256)),
        ToTensor()
    ])

    dataset = ImageDataset(src = "/media/darnok/Nowy/X/datasets/small" , transforms=transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for x in dataloader:
        print("A")
        print("A")




if __name__ == "__main__":
    main()