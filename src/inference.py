#!/usr/bin/env python

# Build in
from pathlib import Path
import argparse

# Config and loggers
import logging
from hydra.utils import instantiate

# ML
import torch

# Own
from helpers.utils import Scaler, load_config, overview
from helpers.viz import init_plot, log_fig, plot_predictions


def main(path) -> None:
    # torch.autograd.set_detect_anomaly(True)
    cfg = load_config(path)
    # Initiate logger
    init_plot(cfg.log.style)
    cfg.log.level = logging.getLevelName(cfg.log.level)
    cfg.log.global_level = logging.getLevelName(cfg.log.global_level)
    logger = logging.getLogger("Inference")
    logger.info("Logs will be saved at %s", Path.cwd())
    # Choosing a device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device -> %s", cfg.device, torch.cuda.get_device_name(0))
    # Inference
    # Initiate dataset
    dataset = instantiate(cfg.dataset, mode="test")
    logger.info(
        "Dataset %s mean: %f std: %f",
        dataset.name,
        *dataset.scaler_info,
    )
    scaler = Scaler(*dataset.scaler_info)
    model = instantiate(
        cfg.model,
        embedding_dict={
            "time": 24 * 7 * 12,
            "day": 7,
            "node": dataset.node,
            "degree": dataset.degrees_max,
        },
        degrees=dataset.degrees,
    )
    e_model = instantiate(
        cfg.estimator,
        embedding_dict={
            "time": 24 * 7 * 12,
            "day": 7,
            "node": dataset.node,
            "degree": dataset.degrees_max,
        },
        degrees=dataset.degrees,
    )
    trainer = instantiate(cfg.trainer, scaler=scaler, model=model, e_model=e_model)
    # Save a performance report
    inference_path = Path.cwd() / "inference" / f"{dataset.name}" / f"{model.name}"
    inference_path.mkdir(exist_ok=True, parents=True)
    if (path / "predictions.pt").is_file():
        preds = torch.load(path / "predictions.pt")
    else:
        preds = trainer.get_predictions(dataset.get_data_loader())
    overview(
        *preds,
        path=inference_path,
        null_val=0,
    )
    for i in range(cfg.dataset.window_size):
        log_fig(
            logger_name="default",
            fig=plot_predictions(*preds, return_fig=True),
            fig_title=f"pred_vs_ground_{i}_step",
            transparent=False,
            path=inference_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Traffic forecasting inference",
        description="Generate a report from a pre-trained model",
        epilog="By Dr. Mourad Lablack",
    )
    parser.add_argument("path", type=str, help="Path to pre-trained folder.")
    args = parser.parse_args()
    main(path=Path(args.path))
