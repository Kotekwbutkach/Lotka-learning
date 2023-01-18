#Importowanie niezbędnych modułów
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

#Podział zbioru na dane treningowe i dane testowe. W każdym wierszu zbioru danych współczynniki układu Lotki-Volterry są inne. 
#Pary punktów w każdym wierszu to rozwiązania układu Lotki-Volterry odpowiednio dla populacji ofiar i populacji drapieżników. 
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
        observed_values = np.array(self.data.loc[id, :2*self.observed_num-1])
        predicted_values = np.array(self.data.loc[id, 2*self.observed_num:(2*(self.observed_num+self.predicted_num)-1)])

        if self.transform:
            observed_values = self.transform(observed_values)
        if self.target_transform:
            predicted_values = self.target_transform(predicted_values)

        return observed_values, predicted_values
