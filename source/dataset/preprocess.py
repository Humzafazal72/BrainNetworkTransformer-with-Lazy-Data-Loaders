import torch
from omegaconf import DictConfig

class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def transform(self, data: torch.Tensor):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor):
        return (data * self.std) + self.mean

def reduce_sample_size(config: DictConfig, *args):
    sz = args[0].shape[0]
    used_sz = int(sz * config.datasz.percentage)
    return [d[:used_sz] for d in args]
