import torch
import json
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MergedDataset(Dataset):
    def __init__(self, csv_file, stats_file, columns_idx):
        df = pd.read_csv(csv_file, sep="\t")
        # Read the min-max statistics from JSON
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        pred_columns = [f'Point{idx}_EtchRate' for idx in columns_idx]

        X_columns = ['Power', 'Pressure', 'Feed', 'Vbias']  # features
        # X_columns = ['Power', 'Pressure']  # features if we had 2 inputs

        # Normalize the data with global min-max
        for column in pred_columns+X_columns:
            df[column] = (df[column] - stats[column]['min']) / (stats[column]['max'] - stats[column]['min'])

        y_columns = pred_columns  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]
        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]


def unscale_min_max(data, column_names, scaling_info):
    unscaled_data = data
    num_columns = data.shape[1]
    #clip the data
    for i, column in enumerate(column_names):
        if i >= num_columns:  # Check if index exceeds the number of columns
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        min_ = scaling_info[column]['min']
        max_ = scaling_info[column]['max']
        negval = data[:,i]<0 #ayto 8a to dioksw
        if np.any(negval):
            print('negative value found before unscale')
        else:
            print('no negative values found before unscale')

        unscaled_data[:, i] = (data[:, i] * (max_-min_)) + min_  # Reverse normalization #edw 8a xrhsimopoihsw to clipped data
        negval2 = unscaled_data[:, i]<0
        if np.any(negval2) < 0:
            print('negative value found after unscale')
        else:
            print('no negative values found after unscale')

    return unscaled_data
