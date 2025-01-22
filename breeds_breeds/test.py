import os

import hydra
import mlflow
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from model import CNN
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from trainer import BreedsTrainer


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    test_dataset = tv.datasets.ImageFolder(
        os.path.join(config.train_config.data_config.test_data_path),
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize(
                    size=(
                        config.train_config.data_config.image_height,
                        config.train_config.data_config.image_width,
                    )
                ),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Нормализация
            ]
        ),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.train_config.data_config.batch_size)

    model = CNN(n_classes=config.train_config.n_classes)
    module = BreedsTrainer.load_from_checkpoint(
        f"{config.model_save_path}/{config.test_config.checkpoint}",
        n_classes=config.train_config.n_classes,
        model=model,
        # lr=config["training"]["lr"],
    )

    trainer = pl.Trainer(
        log_every_n_steps=1, accelerator="auto", devices="auto", default_root_dir="logs"
    )

    results = trainer.test(module, dataloaders=test_dataloader)
    print(results)


if __name__ == "__main__":
    main()
