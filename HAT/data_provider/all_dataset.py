import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path='./dataset/ETT-small', data_path='ETTh1.csv', scale=True, seq_len=384):
        self.seq_len = seq_len
        self.set_type = 0
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[cols]
        if self.scale:
            train_data = df_raw[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        self.data_x = data[border1:border2]

    def __getitem__(self, index):

        item = index % self.data_x.shape[1]
        index = index // self.data_x.shape[1]
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, item]

        return seq_x

    def __len__(self):
        return (len(self.data_x) - self.seq_len + 1) * self.data_x.shape[1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, data_path='ETTm1.csv', scale=True, seq_len=384):

        self.seq_len = seq_len
        self.set_type = 0
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[cols]
        if self.scale:
            train_data = df_raw[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        self.data_x = data[border1:border2]

    def __getitem__(self, index):

        item = index % self.data_x.shape[1]
        index = index // self.data_x.shape[1]
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, item]

        return seq_x

    def __len__(self):
        return (len(self.data_x) - self.seq_len + 1) * self.data_x.shape[1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', scale=True, seq_len=384):
        if root_path == "./dataset/illness":
            self.seq_len = seq_len  # 36
        else:
            self.seq_len = seq_len
        self.set_type = 0
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.scale:
            train_data = df_raw[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values
        self.data_x = data[border1:border2]

    def __getitem__(self, index):

        item = index % self.data_x.shape[1]
        index = index // self.data_x.shape[1]
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, item]

        return seq_x

    def __len__(self):
        return (len(self.data_x) - self.seq_len + 1) * self.data_x.shape[1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, seasonal_patterns='Yearly'):
        self.root_path = './dataset/m4'
        self.pred_len = M4Meta.horizons_map[seasonal_patterns]
        self.seq_len = 2 * self.pred_len
        self.label_len = self.pred_len
        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])

        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])

        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros(self.seq_len)
        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]
        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):] = insample_window
        return insample

    def __len__(self):
        return len(self.timeseries)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class SegLoader(Dataset):
    def __init__(self, root_path, win_size, flag="train"):
        self.flag = flag
        self.win_size = win_size
        self.scaler = StandardScaler()
        if root_path == './dataset/PSM':
            data = pd.read_csv(os.path.join(root_path, 'train.csv'))
            data = data.values[:, 1:]
            data = np.nan_to_num(data)
        elif root_path == './dataset/MSL':
            data = np.load(os.path.join(root_path, "MSL_train.npy"))
        elif root_path == './dataset/SMAP':
            data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        elif root_path == './dataset/SMD':
            data = np.load(os.path.join(root_path, "SMD_train.npy"))
        else:
            data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
            data = data.values[:, :-1]

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.train = data
        print("root_path:", root_path)
        print("train:", self.train.shape)

    def __len__(self):
        len_dateset = 0
        if self.flag == "train":
            len_dateset = self.train.shape[0] - self.win_size + 1
            len_dateset *= self.train.shape[1]
        return len_dateset

    def __getitem__(self, index):
        if self.flag == "train":
            item = index % self.train.shape[1]
            index = index // self.train.shape[1]
            return np.float32(self.train[index:index + self.win_size, item])
        else:
            return None
