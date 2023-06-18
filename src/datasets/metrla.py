# ML
import pandas as pd

# Own
from datasets.base import BaseDataset


class Dataset(BaseDataset):
    def load_data(self):
        self._name = "METRLA"
        self.adj = pd.read_csv(self.data_folder_path / "METRLA" / "W_metrla.csv").values
        data = pd.read_hdf(self.data_folder_path / "METRLA" / "metr-la.h5")
        self.timestamps = data.index
        self.timestamps.freq = self.timestamps.inferred_freq
        self.time_index = self.time_to_idx(self.timestamps, freq="5min")
        self.data = data.values
        self.data_mean, self.data_std = self.data.mean(), self.data.std()
        self.data_min, self.data_max = self.data.min(), self.data.max()
