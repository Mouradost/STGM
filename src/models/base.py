import logging
from pathlib import Path
import torch
import numpy as np


def super_node_pre_hook(_module, args, kwargs):
    # print("super_node_pre_hook", args, kwargs)
    if len(args) > 0:
        args[0] = torch.nn.functional.pad(args[0], (0, 0, 0, 1), value=0)
        for i in range(2, 2 + len(args[2:])):
            args[i] = torch.nn.functional.pad(args[i], (0, 1, 0, 1), value=1)
    if isinstance(kwargs.get("x", None), torch.Tensor):
        kwargs["x"] = torch.nn.functional.pad(kwargs["x"], (0, 0, 0, 1), value=0)
    if isinstance(kwargs.get("adj", None), torch.Tensor):
        kwargs["adj"] = torch.nn.functional.pad(kwargs["adj"], (0, 1, 0, 1), value=1)
    if isinstance(kwargs.get("adj_hat", None), torch.Tensor):
        kwargs["adj_hat"] = torch.nn.functional.pad(
            kwargs["adj_hat"], (0, 1, 0, 1), value=1
        )
    return args, kwargs


def super_node_post_hook(_module, _args, _kwargs, x: torch.Tensor):
    if len(x.shape) > 3:
        return x[..., :-1, :]
    else:
        return x[..., :-1, :-1]


class Embedding(torch.nn.Module):
    def __init__(
        self, embedding_dim: int, embeddings: dict[str, int | None], device: str = "cpu"
    ):
        """
        @params
        embedding_dim: embedding dimension (temporal (T))
        embeddings: dict disribing each embedding depth (keys: time, day, node, degree)
        device: torch.Device (cpu, cuda, tpu)

        @Inputs
        x: tensor of shape (B, C, N, T)
        idx: tensor of shape (B, 2, T)

        @Outputs
        out: tensor of shape (B, C', N, T')
        """
        super().__init__()
        self.logger = logging.getLogger("Embedding Module")
        self.embeddings = embeddings
        if self.embeddings.get("time", None) is not None:
            self.time_embedding = torch.nn.Embedding(
                self.embeddings["time"] + 1, embedding_dim=embedding_dim, device=device
            )
        if self.embeddings.get("day", None) is not None:
            self.day_embedding = torch.nn.Embedding(
                self.embeddings["day"] + 1, embedding_dim=embedding_dim, device=device
            )
        if self.embeddings.get("node", None) is not None:
            self.node_embedding = torch.nn.Embedding(
                self.embeddings["node"] + 1, embedding_dim=embedding_dim, device=device
            )
        if self.embeddings.get("degree", None) is not None:
            self.degree_embedding = torch.nn.Embedding(
                self.embeddings["degree"] + 1,
                embedding_dim=embedding_dim,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        node_ids: torch.Tensor | None = None,
        degrees: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logging.debug(
            "x: %s, idx: %s, node_ids: %s, degrees: %s",
            x.shape,
            idx.shape,
            None if node_ids is None else node_ids.shape,
            None if degrees is None else degrees.shape,
        )
        if self.embeddings.get("time", None) is not None:
            x += self.time_embedding(idx[:, 0]).transpose(1, 2).unsqueeze(2)
            logging.debug("x: %s", x.shape)
        if self.embeddings.get("day", None) is not None:
            x += self.day_embedding(idx[:, 1]).transpose(1, 2).unsqueeze(2)
            logging.debug("x: %s", x.shape)
        if not (self.embeddings.get("node", None) is None or node_ids is None):
            x += (
                self.node_embedding(node_ids)
                .transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(x.shape[0], 1, 1, x.shape[-1])
            )
            logging.debug("x: %s", x.shape)
        if not (self.embeddings.get("degree", None) is None or degrees is None):
            x += (
                self.degree_embedding(degrees)
                .transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(x.shape[0], 1, 1, x.shape[-1])
            )
            logging.debug("x: %s", x.shape)
        return x


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        embedding_dict: dict[str, int | None],
        channels_last: bool = True,
        name: str = "",
        degrees: np.ndarray | None = None,
        use_super_node: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = name
        self.channels_last = channels_last
        self.logger = logging.getLogger(name)
        self.embedding_dict = embedding_dict
        self.hook_handlers = []
        if use_super_node:
            self.hook_handlers.append(
                self.register_forward_pre_hook(super_node_pre_hook, with_kwargs=True)
            )
            self.hook_handlers.append(
                self.register_forward_hook(super_node_post_hook, with_kwargs=True)
            )
            self.embedding_dict.get("node", None)
            if self.embedding_dict.get("node", None) is not None:
                self.embedding_dict["node"] += 1
            if self.embedding_dict.get("degree", None) is not None:
                self.embedding_dict["degree"] = self.embedding_dict.get("node", None)
            if degrees is not None:
                degrees += 1
                degrees = np.concatenate(
                    (degrees, np.array((self.embedding_dict["node"],))), axis=-1
                )

        self.degrees = (
            None
            if degrees is None
            else torch.tensor(degrees, dtype=torch.int, device=device)
        )
        self.node_ids = (
            None
            if self.embedding_dict.get("node", None) is None
            else torch.arange(
                self.embedding_dict["node"], dtype=torch.int, device=device
            )
        )

    def remove_super_node_hooks(self):
        for handle in self.hook_handlers:
            handle.remove()

    def load(self, path: Path = Path.cwd() / "logs" / "model_weights.pth"):
        if path.is_file():
            self.load_state_dict(torch.load(path))
            self.logger.info("Model succesfully loaded")

    def save(self, path: Path = Path.cwd() / "logs" / "model_weights.pth"):
        torch.save(self.state_dict(), path)
        self.logger.info("Model succesfully saved")

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_onnx(self):
        pass
