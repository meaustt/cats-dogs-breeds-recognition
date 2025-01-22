import os

import hydra
import mlflow
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from lightning.pytorch import loggers
from model import CNN
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from trainer import BreedsTrainer


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    train_dataset = tv.datasets.ImageFolder(
        os.path.join(config.train_config.data_config.train_data_path),
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
                ),
            ]
        ),
    )

    val_dataset = tv.datasets.ImageFolder(
        os.path.join(config.train_config.data_config.val_data_path),
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
                ),
            ]
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_config.data_config.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train_config.data_config.batch_size,
        shuffle=False,
    )

    model = CNN(n_classes=config.train_config.n_classes)
    module = BreedsTrainer(
        model=model, n_classes=config.train_config.n_classes, learning_rate=config.train_config.learning_rate
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=config.train_config.training_monitoring,
        dirpath=config.model_save_path,
        filename="model_{val_acc:.2f}",
        save_top_k=1,
        mode=config.train_config.training_monitoring_mode,
    )

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config.train_config.mlflow_logging.experiment_name,
        run_name=config.train_config.mlflow_logging.run_name,
        save_dir=config.train_config.mlflow_logging.mlflow_save_dir,
        tracking_uri=config.train_config.mlflow_logging.tracking_uri,
    )

    tb_logger = loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer(
        max_epochs=config.train_config.num_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=[logger, tb_logger],
        callbacks=[checkpoint_callback],
        default_root_dir="logs",
    )

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_artifact("sandbox.ipynb")
        mlflow.log_param("batch_size", config.train_config.data_config.batch_size)
        mlflow.log_param("lr", config.train_config.learning_rate)
        mlflow.log_param("num_epochs", config.train_config.num_epochs)

    trainer.fit(
        model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    best_model_path = checkpoint_callback.best_model_path
    print(f"Лучшая модель сохранена по пути: {best_model_path}")


if __name__ == "__main__":
    main()
