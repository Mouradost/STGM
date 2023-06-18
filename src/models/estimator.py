import torch
import numpy as np

from models.base import BaseModel, Embedding


class Model(BaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dict: dict[str, int | None],
        bias: bool = False,
        channels_last: bool = True,
        degrees: np.ndarray | None = None,
        name: str = "Estimator",
        device: str = "cpu",
        *args,
        **kwargs
    ) -> None:
        """
        @params
        in_channels: number of input channels (C)
        hidden_channels: number of itermediate hidden channels (C')
        bias: if the biases should be added (please set it to False when using layer norm)
        channels_last: if the channels are situated in the last dim
        name: The name of the model
        device: torch.Device (cpu, cuda, tpu)

        @Inputs
        x: tensor of shape (B, T, N, C)

        @Outputs
        out: tensor of shape (B, N, N)
        """
        super().__init__(
            name=name,
            channels_last=channels_last,
            degrees=degrees,
            use_super_node=False,
            embedding_dict=embedding_dict,
            device=device,
        )

        self.fc_in = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )

        self.embedding = Embedding(
            embedding_dim=hidden_channels,
            embeddings=self.embedding_dict,
            device=device,
        )

        self.k = self._block(
            in_channels=hidden_channels, out_channels=1, bias=bias, device=device
        )
        self.q = self._block(
            in_channels=hidden_channels, out_channels=1, bias=bias, device=device
        )
        # self.act = torch.nn.LeakyReLU()
        self.act = torch.nn.Sigmoid()

    def _block(self, in_channels: int, out_channels: int, bias: bool, device: str):
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                bias=bias,
                device=device,
            ),
            torch.nn.BatchNorm2d(out_channels, device=device),
            torch.nn.ELU(),
        )

    def forward(
        self, x: torch.Tensor, idx: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if self.channels_last:
            x = x.transpose(1, -1)
        x = self.fc_in(x)  # B, C', N, T
        x = self.embedding(
            x, idx, degrees=self.degrees, node_ids=self.node_ids
        )  # B, C', N, T
        k = self.k(x)  # B, 1, N, T
        q = self.q(x)  # B, 1, N, T
        return self.act(torch.einsum("BCNT, BCnT -> BCNn", k, q)).squeeze(1)  # B, N, N
