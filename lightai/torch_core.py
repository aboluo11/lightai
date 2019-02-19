import numpy as np

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import BatchSampler, RandomSampler, Sampler, SequentialSampler, DataLoader, SubsetRandomSampler


def T(x: np.ndarray, cuda=True):
    if isinstance(x, (list, tuple)):
        return [T(each) for each in x]
    x = np.ascontiguousarray(x)
    if x.dtype in (np.uint8, np.int8, np.int16, np.int32, np.int64):
        x = torch.from_numpy(x.astype(np.int64))
    elif x.dtype in (np.float32, np.float64):
        x = torch.from_numpy(x.astype(np.float32))
    if cuda:
        x = x.pin_memory().cuda(non_blocking=True)
    return x
