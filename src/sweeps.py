#!/usr/bin/env python

# Build in
from pathlib import Path

# ML
import torch

# Config and loggers
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb

# Own
from helpers.utils import Scaler


@hydra.main(
    version_base=None,
    config_path=(Path.cwd() / "config").as_posix(),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # Choosing a device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Training
    # Initiate dataset
    train_dataset = instantiate(cfg.dataset, mode="train")
    val_dataset = instantiate(cfg.dataset, mode="val")
    # Initiate the scaler
    scaler = Scaler(*train_dataset.get_scaler_info())
    # Initiate the model
    model = instantiate(cfg.model, degrees=train_dataset.degree)
    # Setup a logger
    wandb.init(
        project=cfg.log.project,
        settings=wandb.Settings(start_method="thread"),
    )
    # Initiate trainer
    trainer = instantiate(
        cfg.trainer, model=model, scaler=scaler, logger_name=cfg.log.logger
    )
    # Train
    trainer.train(train_dataset.get_data_loader(), val_dataset.get_data_loader())


if __name__ == "__main__":
    # run sweep with wandb sweep ./config/sweepers/[YOUR SWEEP OF CHOICE]
    # Then follow the wandb instructions
    main()
