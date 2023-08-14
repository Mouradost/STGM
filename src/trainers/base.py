# Build-in
import logging
from pathlib import Path

# Log
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# ML
import torch

# Own
from helpers.utils import Scaler
from helpers.metrics import MAE


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        scaler: Scaler,
        clip: float = 1.0,
        epochs: int = 1000,
        lr: float = 1e-3,
        loss_fn=MAE,
        verbose: bool = True,
        use_amp: bool = False,
        model_pred_single: bool = False,  # if the model only output one timesteps
        log_level=logging.CRITICAL,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(log_level)
        self.model_pred_single = model_pred_single
        self.epochs = epochs
        self.model = model
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.device = device
        self.verbose = verbose
        self.use_amp = use_amp
        self.clip = clip
        self.grad_sclaer = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-5,
        )

        self.metrics = {
            "train/loss": 0.0,
            "val/loss": 0.0,
        }
        self.history = {
            "epoch": 0,
            "train/loss": [],
            "val/loss": [],
            "train/clip": [],
        }
        self.log_step = self.epochs // 2 if self.epochs // 2 != 0 else 1

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def train_step(self):
        raise (NotImplemented)

    def train(self):
        raise (NotImplemented)

    def val_step(self):
        raise (NotImplemented)

    def validate(self):
        raise (NotImplemented)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        idx: torch.Tensor | None = None,
        adj: torch.Tensor | None = None,
        sim: torch.Tensor | None = None,
    ):
        if self.model_pred_single:
            y = y[:, -1:]
        return self.loss_fn(
            pred=self.scaler.reverse(
                self.model(
                    x=self.scaler.norm(x),
                    idx=idx,
                    adj=adj,
                    adj_hat=sim,
                )
            ),
            target=y,
            null_val=0,
        )

    def log(self, epoch: int):
        self.history["epoch"] += 1
        self.history["train/loss"].append(self.metrics["train/loss"])
        self.history["val/loss"].append(self.metrics["val/loss"])
        metrics_str = ""
        for key, value in self.metrics.items():
            metrics_str += f"{key}: {value / self.history['epoch']:.2E} "
        self.logger.info("Epoch: %i -> %s", epoch, metrics_str)

    def save_model(self):
        self.model.save(Path.cwd() / f"{self.model.name}_model.pth")

    def load_model(self):
        self.model.load(Path.cwd() / f"{self.model.name}_model.pth")

    @torch.no_grad()
    def get_predictions(
        self, test_data: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_grounds, all_predictions = [], []
        loop = tqdm(test_data, total=len(test_data), desc="Generating predictions: ")
        with logging_redirect_tqdm():
            for idx, adj, sim, x, y in loop:
                idx, adj, sim, x = (
                    idx.to(self.device),
                    adj.to(self.device),
                    sim.to(self.device),
                    x.to(self.device),
                )
                prediction = self.scaler.reverse(
                    self.model(x=self.scaler.norm(x), adj=adj, adj_hat=sim, idx=idx)
                    .cpu()
                    .detach()
                )
                all_grounds.append(y.detach())
                all_predictions.append(prediction)
        return torch.cat(all_predictions), torch.cat(all_grounds)
