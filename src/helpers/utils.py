# Build in
from pathlib import Path
import logging

# ML
import torch
import wandb

# Config
from omegaconf import OmegaConf
import wandb

# Own
from helpers.metrics import summary


class Scaler:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def norm(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std

    def reverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std + self.mean


def resolve_tuple(*args):
    return tuple(args)


def init():
    if not OmegaConf.has_resolver("tuple"):
        OmegaConf.register_new_resolver("tuple", resolve_tuple)


def load_config(path: Path, display: bool = False, resolve: bool = True):
    if (path / "config.yaml").exists():
        cfg = OmegaConf.load(path / "config.yaml")
    else:
        cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    if display:
        print(OmegaConf.to_yaml(cfg, resolve=resolve))
    return cfg


def setup_logger(cfg):
    # Initializing the logger
    logging.basicConfig(
        filename=Path(cfg.paths.log) / "run.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %I:%M:%S %p",
        encoding="utf-8",
        level=cfg.log.global_level,
    )
    if cfg.log.logger == "wandb":
        wandb.init(
            project=cfg.log.project,
            name=f"{cfg.model.name}",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            # settings=wandb.Settings(start_method="thread"),
        )
    elif cfg.log.logger == "tensorboard":
        pass
    else:
        pass


def post_logging(logger_name: str):
    if logger_name == "wandb":
        wandb.finish()
    elif logger_name == "tensorboard":
        pass
    else:
        pass


def overview(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    path: Path | None = None,
    null_val: float = torch.nan,
    dim: tuple = (0, 2, 3),
) -> str:
    results = summary(
        target=targets,
        pred=predictions,
        suffix="Real",
        dim=dim,
        null_val=null_val,
    )[0]
    if path is not None:
        with (path / "performance.txt").open(mode="w") as f:
            f.write(results)

    logging.info("\n%s", results)

    return results
