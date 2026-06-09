from eidos.models import ViT
from eidos.transforms import LoadImage
from eidos.datasets import ImageDataset
from eidos.regularization import SIGReg, GramReg

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize


def train_step(dataloader:DataLoader, model:ViT, opt:AdamW, criterions:dict, weights=None):
    
    if weights is None:
        weights = {k: 1.0 for k in criterions.keys()}

    epoch_losses = {k: 0.0 for k in criterions.keys()}
    total_loss_accum = 0.0

    for x, neighbors in dataloader:

        opt.zero_grad()

        embeddings = model(x)  # [B, N, D]

        batch_losses = {}

        # -----------------------------
        # compute all losses
        # -----------------------------
        for name, crit in criterions.items():

            if name == "GramReg":
                out = crit(embeddings, neighbors)
                loss = out["total"]

            else:
                loss = crit(embeddings)

            batch_losses[name] = loss

        # -----------------------------
        # weighted sum
        # -----------------------------
        total_loss = 0.0
        for name, loss in batch_losses.items():
            w = weights.get(name, 1.0)
            total_loss = total_loss + w * loss

            epoch_losses[name] += loss.item()

        total_loss.backward()
        opt.step()

        total_loss_accum += total_loss.item()
        steps += 1

    # -----------------------------
    # finalize logs
    # -----------------------------
    for k in epoch_losses:
        epoch_losses[k] /= steps

    epoch_losses["total"] = total_loss_accum / steps

    return epoch_losses

def main(epochs:int):
    
    transforms = Compose([
        LoadImage(),
        Resize((256,256)),
        ToTensor()
    ])

    dataset = ImageDataset(src = "/media/darnok/Nowy/X/datasets/small" , transforms=transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    model = ViT()
    opt = AdamW(params = model.parameters(), lr=3e-4)

    criterions = {
        "SigReg": SIGReg(),
        "GramReg":GramReg()
    }

    model.train() 
    for epoch in range(epochs):
        loss = train_step(dataloader=dataloader, model=model, opt=opt, criterions=criterions)
        print(f"Epoch: {epoch} | Loss: {loss}")

if __name__ == "__main__":
    main(epochs=100)