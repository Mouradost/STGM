#!/usr/bin/env python

# Build in
from pathlib import Path

# Config and loggers
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# ML
import torch

# Own
from helpers.utils import Scaler, setup_logger, overview
from helpers.viz import init_plot, log_fig, plot_history, plot_predictions


@hydra.main(
    version_base=None,
    config_path=(Path.cwd() / "config").as_posix(),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # torch.autograd.set_detect_anomaly(True)
    # Initiate logger
    init_plot(cfg.log.style)
    cfg.log.level = logging.getLevelName(cfg.log.level)
    cfg.log.global_level = logging.getLevelName(cfg.log.global_level)
    setup_logger(cfg)
    logger = logging.getLogger("Run")
    logger.info("Logs will be saved at %s", Path.cwd())
    # Choosing a device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device -> %s", cfg.device, torch.cuda.get_device_name(0))
    # Config overview
    with (Path.cwd() / "config.yaml").open(mode="w") as f:
        OmegaConf.save(cfg, f)
    # Training
    # Initiate dataset
    train_dataset = instantiate(cfg.dataset, mode="train")
    val_dataset = instantiate(cfg.dataset, mode="val")
    logger.info(
        "Dataset %s mean: %f std: %f",
        train_dataset.name,
        *train_dataset.scaler_info,
    )
    scaler = Scaler(*train_dataset.scaler_info)
    model = instantiate(
        cfg.model,
        embedding_dict={
            "time": 24 * 7 * 12,
            "day": 7,
            "node": train_dataset.node,
            "degree": train_dataset.degrees_max,
        },
        degrees=train_dataset.degrees,
    )
    e_model = instantiate(
        cfg.estimator,
        embedding_dict={
            "time": 24 * 7 * 12,
            "day": 7,
            "node": train_dataset.node,
            "degree": train_dataset.degrees_max,
        },
        degrees=train_dataset.degrees,
    )
    trainer = instantiate(cfg.trainer, scaler=scaler, model=model, e_model=e_model)
    # Train
    log_fig(
        logger_name=cfg.log.logger,
        fig=plot_history(
            trainer.train(
                train_data=train_dataset.get_data_loader(),
                val_data=val_dataset.get_data_loader(),
            )
        ),
        fig_title="train_history",
        transparent=False,
    )

    # Save model
    trainer.save_model()

    # Save a performance report
    preds = trainer.get_predictions(
        instantiate(cfg.dataset, mode="test").get_data_loader()
    )
    torch.save(preds, Path.cwd() / "predictions.pt")

    overview(
        *preds,
        path=Path.cwd(),
        null_val=0,
    )
    for i in range(cfg.dataset.window_size):
        log_fig(
            logger_name="default",
            fig=plot_predictions(*preds, return_fig=True),
            fig_title=f"pred_vs_ground_{i}_step",
            transparent=False,
            path=Path.cwd(),
        )


if __name__ == "__main__":
    main()
