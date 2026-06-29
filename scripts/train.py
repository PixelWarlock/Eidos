from __future__ import annotations
import yaml
import argparse
import lightning as L
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from eidos.dataset import EidosDataset
from eidos.model import EidosModule, Eidos

from albumentations.augmentations.
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

def _get_transforms_from_config(config):
    return ...

def _dataset(config: dict[str, Any], split: str) -> EidosDataset:
    if split == 'train':
        shared_transforms = _get_transforms_from_config(config=config)
        
    return EidosDataset(
        img_dir=config[f'{split}_images'],
        mask_dir=config[f'{split}_masks'],
        shared_transforms=shared_transforms
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    L.seed_everything(42, workers=True)

    train_data = _dataset(config["data"], "train")
    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=1)

    module = EidosModule(
        Eidos(**config["model"]),
        regularizers=config["regularizers"],
        learning_rate=config["train"]["learning_rate"],
    )
    output = config["train"].get("output_dir", "outputs")
    checkpoint = ModelCheckpoint(output / "checkpoints", monitor="val/loss", save_last=True)
    trainer = L.Trainer(
        max_epochs=config["train"]["epochs"],
        callbacks=[checkpoint],
        logger=[TensorBoardLogger(output, name="tensorboard"), CSVLogger(output, name="csv")],
        default_root_dir=output,
    )
    trainer.fit(module, train_loader) #valid_loader


if __name__ == "__main__":
    main()
