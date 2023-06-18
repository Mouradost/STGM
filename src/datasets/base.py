import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch

from helpers.dtw import parallel_dtw as dtw


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str = "PEMSBAY",
        data_folder_path: Path = Path.home() / "Documents" / "Datasets",
        mode: str = "test",
        window_size: int = 12,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 128,
        nb_worker: int = 0,
        r: float = 1.0,
        log_level: int = logging.INFO,
    ):
        self._name = name

        self.logger = logging.getLogger("Dataset")
        self.logger.setLevel(log_level)
        if not isinstance(data_folder_path, Path):
            data_folder_path = Path(data_folder_path)
        self.window_size = window_size
        self.mode = mode.lower()
        self.batch_size = batch_size
        self.num_workers = nb_worker

        self.data_folder_path = data_folder_path
        self.r = r
        self.sim = None

        self.load_data()
        self.adj = self.normalize_adj(self.adj, self.r)

        if self.mode == "train":
            last_index = int((train_ratio - val_ratio) * self.data.shape[0])
            self.data = self.data[:last_index]
            self.time_index = self.time_index[:last_index]
            self.timestamps = self.timestamps[:last_index]
        elif self.mode == "val":
            first_index = int((train_ratio - val_ratio) * self.data.shape[0])
            last_index = int(train_ratio * self.data.shape[0])
            self.data = self.data[first_index:last_index]
            self.time_index = self.time_index[first_index:last_index]
            self.timestamps = self.timestamps[first_index:last_index]
        else:
            first_index = int(train_ratio * self.data.shape[0])
            self.data = self.data[first_index:]
            self.time_index = self.time_index[first_index:]
            self.timestamps = self.timestamps[first_index:]
        self.total_size = self.data.shape[0] - 2 * self.window_size
        self.nb_nodes = self.data.shape[-1]
        self.graph = nx.from_numpy_array(self.adj)
        self.logger.info(
            "Dataset %s (%s) length: %s", self.name, self.mode, self.__len__()
        )
        if self.sim is None:
            self.logger.warn(
                "No pre-calculated contribution matrix was provided ! You will experience Heavy CPU load and Slow data loading..."
            )
        else:
            self.sim = self.normalize_sim(self.sim, self.r)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx: int):
        return (
            torch.tensor(
                self.time_index[idx : idx + self.window_size],
                dtype=torch.int,
            ).T,
            torch.tensor(self.adj, dtype=torch.float),
            torch.tensor(self.dtw(idx), dtype=torch.float),
            torch.tensor(
                self.data[
                    idx : idx + self.window_size,
                    :,
                ],
                dtype=torch.float,
            ).unsqueeze(-1),
            torch.tensor(
                self.data[
                    idx + self.window_size : idx + 2 * self.window_size,
                    :,
                ],
                dtype=torch.float,
            ).unsqueeze(-1),
        )

    def get_sample(self, idx: int = 0, with_timestamps: bool = False):
        if with_timestamps:
            return (
                self.timestamps[idx : idx + self.window_size],
                torch.tensor(
                    self.time_index[idx : idx + self.window_size],
                    dtype=torch.int,
                ).T.unsqueeze(0),
                torch.tensor(self.adj, dtype=torch.float).unsqueeze(0),
                torch.tensor(self.dtw(idx), dtype=torch.float).unsqueeze(0),
                torch.tensor(
                    self.data[
                        idx : idx + self.window_size,
                        :,
                    ],
                    dtype=torch.float,
                )
                .unsqueeze(0)
                .unsqueeze(-1),
                torch.tensor(
                    self.data[
                        idx + self.window_size : idx + 2 * self.window_size,
                        :,
                    ],
                    dtype=torch.float,
                )
                .unsqueeze(0)
                .unsqueeze(-1),
            )
        else:
            return (
                torch.tensor(
                    self.time_index[idx : idx + self.window_size],
                    dtype=torch.int,
                ).T.unsqueeze(0),
                torch.tensor(self.adj, dtype=torch.float).unsqueeze(0),
                torch.tensor(self.dtw(idx), dtype=torch.float).unsqueeze(0),
                torch.tensor(
                    self.data[idx : idx + self.window_size, :],
                    dtype=torch.float,
                )
                .unsqueeze(0)
                .unsqueeze(-1),
                torch.tensor(
                    self.data[
                        idx + self.window_size : idx + 2 * self.window_size,
                        :,
                    ],
                    dtype=torch.float,
                )
                .unsqueeze(0)
                .unsqueeze(-1),
            )

    def dtw(self, idx: int):
        return self.adj
        if self.sim is not None:
            if len(self.sim.shape) > 2:
                return self.sim[idx]
            return self.sim
        return self.normalize_sim(
            dtw(
                self.data[idx : idx + self.window_size].T,
                self.data[idx + self.window_size : idx + 2 * self.window_size].T,
            ),
            self.r,
        )

    def load_data(self):
        pass

    @staticmethod
    def normalize_adj(adj: np.ndarray, r: float = 1.0):
        a = adj.copy()
        a[a != 0] = np.exp(-((a[a != 0] / (a.mean() * r)) ** 2))
        a[a <= 0.1] = 0
        return a

    @staticmethod
    def normalize_sim(sim: np.ndarray, r: float = 1.0):
        s = sim.copy()
        s[s != 0] = 1 - np.exp(-((s[s != 0] / (s.std() * r)) ** 2))
        return s

    @staticmethod
    def time_to_idx(time_indexes, freq: str = "5min"):
        hashmap = {
            t: i for i, t in enumerate(pd.date_range("00:00", "23:55", freq=freq).time)
        }
        results = []
        for timestamp in time_indexes:
            results.append([hashmap[timestamp.time()], timestamp.weekday()])
        return np.array(results, dtype=np.int32)

    @property
    def node(self):
        return self.nb_nodes

    @property
    def degrees(self) -> np.ndarray:
        return np.array(self.graph.degree)[:, 1]

    @property
    def degrees_max(self) -> np.ndarray:
        return self.degrees.max()

    @property
    def scaler_info(self):
        return self.data_mean, self.data_std

    @property
    def scaler_info_mm(self):
        return self.data_min, self.data_max

    @property
    def name(self):
        """
        Return the name of the database
        """
        return self._name

    def get_data_loader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.mode == "train",
            drop_last=False,
        )
