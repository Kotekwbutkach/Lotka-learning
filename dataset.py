import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, datafile, observed_num, predicted_num, transform=None, target_transform=None):
        self.data = pd.read_csv(datafile, sep=";", header=None)
        self.observed_num = observed_num  # N
        self.predicted_num = predicted_num  # k
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        observed_values = self.data.loc[id, :self.observed_num]
        predicted_values = self.data.loc[id, self.observed_num:(self.observed_num+self.predicted_num)]

        if self.transform:
            observed_values = self.transform(observed_values)
        if self.target_transform:
            predicted_values = self.target_transform(predicted_values)

        return observed_values, predicted_values
