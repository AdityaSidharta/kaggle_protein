import torch
from torch.utils.data import Dataset

from utils.common import *
from utils.interface import *


class ProteinTrainDataset(Dataset):
    def __init__(self, train_df, path, device):
        self.train_df = train_df
        self.path = path
        self.device = device

        self.fn_list = train_df.Id.values.tolist()
        self.tgt_list = [x.split() for x in train_df.Target.values.tolist()]
        self.tgt_list = [list(map(int, x)) for x in self.tgt_list]

        self.n_class = len(class_dict)
        self.n_obs = len(self.fn_list)

    def __getitem__(self, idx):
        img_fn = self.fn_list[idx]
        target = self.tgt_list[idx]

        img_array = open_rgby(self.path, img_fn).astype(float)
        target_array = np.eye(self.n_class)[target].sum(axis=0).astype(float)

        return torch.from_numpy(img_array).to(self.device), torch.from_numpy(target_array).to(self.device)

    def __len__(self):
        return self.n_obs


class ProteinTestDataset(Dataset):
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.fn_list = [x.split('.')[0] for x in os.listdir(path)]
        self.n_class = len(class_dict)
        self.n_obs = len(self.fn_list)

    def __getitem__(self, idx):
        img_fn = self.fn_list[idx]
        img_array = open_rgby(self.path, img_fn).astype(float)
        return torch.from_numpy(img_array).to(self.device)

    def __len__(self):
        return self.n_obs