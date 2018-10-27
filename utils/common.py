from utils.envs import *
from skimage.io import imread
from sklearn.model_selection import train_test_split
import numpy as np
import math
import torch
from itertools import tee

def get_train_fn():
    return [x[:36] for x in os.listdir(train_path)]


def get_test_fn():
    return [x[:36] for x in os.listdir(test_path)]


def open_rgby(path, file_name):
    colors = ["red", "green", "blue", "yellow"]
    img = [
        imread(os.path.join(path, "{}_{}.png".format(file_name, color))).astype(
            np.float32
        )
        / 255
        for color in colors
    ]
    return np.stack(img, axis=-1)


def split_dev_val(train_df, val_size, random_state = 0):
    if random_state:
        dev_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    else:
        dev_df, val_df = train_test_split(train_df, test_size=val_size)
    return dev_df, val_df

def get_batch_info(dataloader):
    n_obs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    n_batch_per_epoch = math.ceil(n_obs / float(batch_size))
    return n_obs, batch_size, n_batch_per_epoch

def img2tensor(img_array, device):
    img_array = img_array.transpose((2, 0, 1))
    return torch.from_numpy(img_array).float().to(device)


def to_list(x):
    return x if type(x) == list else [x]



def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
