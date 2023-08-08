import os
import torch
import torch.utils.data as data
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MATPaddedDataset(data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [f for f in os.listdir(data_folder) if f.endswith('.mat')]

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        mat_data = sio.loadmat(os.path.join(self.data_folder, mat_file))
        volume = mat_data['new_data']  # Assuming the data is stored under the key 'T00'

        # Normalize the data to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Convert to torch tensor and add a channel dimension (assuming grayscale images)
        volume = torch.tensor(volume).unsqueeze(0).float()
        return volume

    def __len__(self):
        return len(self.file_list)




# Assuming you have a folder 'data_folder' containing the .mat files
data_folder = 'C:\\Users\ericc\PycharmProjects\VAE_10000\inputMATfiles'


# Assuming you already know the maximum depth of all volumes in the dataset

# Create the MATPaddedDataset and DataLoader
batch_size = 3
dataset = MATPaddedDataset(data_folder)
data_samples = dataset.__len__()
train_samples = round(data_samples * 0.8)
validation_samples = round(data_samples * 0.1)
test_samples = data_samples - (train_samples + validation_samples)
train_Set, val_Set, test_Set = torch.utils.data.random_split(
        dataset, [train_samples, validation_samples, test_samples], generator=torch.Generator().manual_seed(85210))

train_loader = DataLoader(train_Set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(val_Set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_Set, batch_size=batch_size, shuffle=True, num_workers=0)
