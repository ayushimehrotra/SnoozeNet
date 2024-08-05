import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import scipy.signal as signal

from utils import clear_mem


firstk = 5

def stft(raw_data, window_size=250, overlap=50, nfft=250):
    '''
        raw_data = (number_of_patients, ((number_of_epochs, len_of_epochs), (number_of_epochs))
    '''
    # Parameters for STFT
    sampling_rate = 100  # 100 Hz
    nperseg = window_size
    noverlap = overlap

    # Perform STFT on the entire dataset
    stft_data = []
    # Shape: (number_of_patients, ((number_of_epochs, len_of_epochs), (number_of_epochs))
    for eeg_data, preds in raw_data:
        tranformed_data = []
        for epoch in eeg_data:
            raw_epoch = [x[0] for x in epoch]
            f, t, Zxx = signal.stft(raw_epoch, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            tranformed_data.append(np.abs(Zxx))

        # print(tranformed_data.shape, preds.shape)
        stft_data.append([tranformed_data, preds])

    stft_data = np.array(stft_data, dtype=object)
    return stft_data

def load_data(folder_path):
    npz_files = [os.path.join(folder_path, file)
                 for file in os.listdir(folder_path) if file.endswith('.npz')]

    data = []
    for file in npz_files:
        with np.load(file, allow_pickle=True) as f:
            data.append([f['x'], f['y']])
    del npz_files
    clear_mem()
    return np.array(data, dtype=object)


class EEGDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = np.array(raw_data, dtype=object)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return (torch.tensor(np.array(self.raw_data[idx][0])),
        torch.tensor(np.array(self.raw_data[idx][1])))


def collate_fn(batch):
    batch = np.array(batch, dtype=object)
    max_length = max([x.size(0) for x, y in batch])

    padded_x_batch = []
    padded_y_batch = []
    for x, y in batch:
        pad_size = max_length - x.size(0)
        # Pad the tensors
        padded_x = F.pad(x, (0, 0, 0, 0, pad_size, 0))
        padded_y = F.pad(y, (0, pad_size))

        padded_x = x[:firstk, :, :]
        padded_y = y[:firstk]

        padded_x_batch.append(padded_x)
        padded_y_batch.append(padded_y)

        del padded_x, padded_y
        clear_mem()

    # Stack the padded sequences
    padded_x_batch = torch.stack(padded_x_batch)
    padded_y_batch = torch.stack(padded_y_batch)


    return padded_x_batch, padded_y_batch


def create_dataloaders(raw_data_train, raw_data_test):
    '''
        raw_data_train, raw_data_test: (number of patients, (eeg data, predictions))
    '''
    raw_data_train, raw_data_test = stft(raw_data_train, 78, 40, 78), stft(raw_data_test, 78, 40, 78)
    data_train = EEGDataset(raw_data_train)
    data_test = EEGDataset(raw_data_test)
    train_dataloader = DataLoader(data_train, batch_size=32,
                                  shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(data_test, batch_size=1,
                                 shuffle=False, collate_fn=collate_fn)
    return train_dataloader, test_dataloader
