# Build-in
import logging

# Log
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# ML
import torch

# Own
from helpers.utils import Scaler
from helpers.metrics import MAE
from trainers.base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        e_model: torch.nn.Module,
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
        super().__init__(
            model=model,
            scaler=scaler,
            clip=clip,
            epochs=epochs,
            lr=lr,
            loss_fn=loss_fn,
            verbose=verbose,
            use_amp=use_amp,
            model_pred_single=model_pred_single,
            log_level=log_level,
            device=device,
            *args,
            **kwargs,
        )
        self.e_model = e_model

        self.e_opt = torch.optim.AdamW(
            self.e_model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-5,
        )

        self.metrics = {
            "train/loss": 0.0,
            "val/loss": 0.0,
            "train/e_loss": 0.0,
            "val/e_loss": 0.0,
        }
        self.history = {
            "epoch": 0,
            "train/loss": [],
            "val/loss": [],
            "train/e_loss": [],
            "val/e_loss": [],
            "train/clip": [],
        }
        self.log_step = self.epochs // 2 if self.epochs // 2 != 0 else 1

    def val_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sim: torch.Tensor,
        idx: torch.Tensor | None = None,
        adj: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        # Compute the loss.
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.use_amp
        ):
            loss, e_loss = self.loss(x=x, y=y, idx=idx, adj=adj, sim=sim)
        return loss.item(), e_loss.item()

    @torch.no_grad()
    def validate(self, val_data: torch.utils.data.DataLoader):
        for i, (idx, adj, sim, x, y) in enumerate(val_data):
            idx, adj, sim, x, y = (
                idx.to(self.device),
                adj.to(self.device),
                sim.to(self.device),
                x.to(self.device),
                y.to(self.device),
            )
            # Validation step
            loss, e_loss = self.val_step(x=x, y=y, idx=idx, adj=adj, sim=sim)

            self.metrics["val/loss"] += loss / (i + 1)
            self.metrics["val/e_loss"] += e_loss / (i + 1)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sim: torch.Tensor,
        idx: torch.Tensor | None = None,
        adj: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        loss, e_loss = self.loss(x=x, y=y, idx=idx, adj=adj, sim=sim)

        self.e_opt.zero_grad(set_to_none=True)
        e_loss.backward()
        self.e_opt.step()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return loss.item(), e_loss.item()

    def train(
        self,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader = None,
    ):
        self.model.train()
        self.e_model.train()
        epoch_loop = tqdm(
            range(self.epochs),
            desc=f"Epoch 0/{self.epochs} train",
            leave=True,
            disable=not self.verbose,
        )
        with logging_redirect_tqdm():
            for epoch in epoch_loop:
                self.metrics["train/loss"] = 0
                self.metrics["val/loss"] = 0
                self.metrics["train/e_loss"] = 0
                self.metrics["val/e_loss"] = 0
                for i, (idx, adj, sim, x, y) in enumerate(train_data):
                    idx, adj, sim, x, y = (
                        idx.to(self.device),
                        adj.to(self.device),
                        sim.to(self.device),
                        x.to(self.device),
                        y.to(self.device),
                    )

                    # Train step
                    loss, e_loss = self.train_step(x=x, y=y, idx=idx, adj=adj, sim=sim)

                    self.metrics["train/loss"] += loss / (i + 1)
                    self.metrics["train/e_loss"] += e_loss / (i + 1)

                # Validation
                if val_data is not None:
                    self.model.eval()
                    self.validate(val_data=val_data)
                    self.model.train()

                # Log metrics
                self.log(epoch=epoch + 1)
                epoch_loop.set_postfix(
                    loss=self.metrics["train/loss"],
                    loss_val=self.metrics["val/loss"],
                    e_loss=self.metrics["train/e_loss"],
                    e_loss_val=self.metrics["val/e_loss"],
                )
                epoch_loop.set_description_str(f"Epoch {epoch + 1}/{self.epochs} train")

        return self.history

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sim: torch.Tensor,
        idx: torch.Tensor | None = None,
        adj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_pred_single:
            y = y[:, -1:]

        contrib = self.e_model(x=self.scaler.norm(x), idx=idx)
        e_loss = self.loss_fn(contrib, sim, dim=(0, 1, 2))
        loss = self.loss_fn(
            pred=self.scaler.reverse(
                self.model(
                    x=self.scaler.norm(x),
                    idx=idx,
                    adj=adj,
                    adj_hat=contrib.detach(),
                )
            ),
            target=y,
            null_val=0,
        )
        return loss, e_loss

    @torch.no_grad()
    def get_predictions(
        self, test_data: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_grounds, all_predictions = [], []
        loop = tqdm(test_data, total=len(test_data), desc="Generating predictions: ")
        with logging_redirect_tqdm():
            for idx, adj, _, x, y in loop:
                idx, adj, x = (
                    idx.to(self.device),
                    adj.to(self.device),
                    x.to(self.device),
                )
                prediction = self.scaler.reverse(
                    self.model(
                        x=self.scaler.norm(x),
                        adj=adj,
                        adj_hat=self.e_model(x=self.scaler.norm(x), idx=idx),
                        idx=idx,
                    )
                    .cpu()
                    .detach()
                )
                all_grounds.append(y.detach())
                all_predictions.append(prediction)
        return torch.cat(all_predictions), torch.cat(all_grounds)
