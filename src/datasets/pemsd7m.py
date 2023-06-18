# ML
import pandas as pd

# Own
from datasets.base import BaseDataset


class Dataset(BaseDataset):
    def load_data(self):
        self._name = "PEMSD7M"
        self.adj = pd.read_csv(self.data_folder_path / "PEMSD7M" / "W_228.csv").values
        data = pd.read_csv(self.data_folder_path / "PEMSD7M" / "V_228.csv")
        data.time = pd.to_datetime(data.time, format="%Y-%m-%d %H:%M:%S")
        data = data.set_index("time")
        self.timestamps = data.index
        self.time_index = self.time_to_idx(self.timestamps, freq="5min")
        self.data = data.values
        self.data_mean, self.data_std = self.data.mean(), self.data.std()
        self.data_min, self.data_max = self.data.min(), self.data.max()
