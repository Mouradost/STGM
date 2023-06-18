# ML
import torch

# Log
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Own
from trainers.base import BaseTrainer


class Trainer(BaseTrainer):
    def val_step(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        idx: torch.Tensor | None=None, 
        adj: torch.Tensor | None=None, 
        sim: torch.Tensor | None=None
    ) -> float:
        # Compute the loss.
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.use_amp
        ):
            loss = self.loss(x=x, y=y, idx=idx, adj=adj, sim=sim)
        return loss.item()

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
            self.metrics["val/loss"] += self.val_step(x=x, y=y, idx=idx, adj=adj, sim=sim) / (i + 1)

    def train_step(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        idx: torch.Tensor | None=None, 
        adj: torch.Tensor | None=None, 
        sim: torch.Tensor | None=None
    ) -> float:
        # Compute the loss.
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.use_amp
        ):
            loss = self.loss(x=x, y=y, idx=idx, adj=adj, sim=sim)
        # Backward pass: compute gradient of the loss with respect to parameters
        self.grad_sclaer.scale(loss).backward()
        self.grad_sclaer.step(self.opt)
        self.grad_sclaer.update()
        # Before the backward pass, zero all of the network gradients
        self.opt.zero_grad(set_to_none=True)
        return loss.item()

    def train(
        self,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader = None,
    ):
        self.model.train()
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
                for i, (idx, adj, sim, x, y) in enumerate(train_data):
                    idx, adj, sim, x, y = (
                        idx.to(self.device),
                        adj.to(self.device),
                        sim.to(self.device),
                        x.to(self.device),
                        y.to(self.device),
                    )

                    # Train step
                    self.metrics["train/loss"] += self.train_step(x=x, y=y, idx=idx, adj=adj, sim=sim) / (
                        i + 1
                    )

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
                )
                epoch_loop.set_description_str(f"Epoch {epoch + 1}/{self.epochs} train")

        return self.history
