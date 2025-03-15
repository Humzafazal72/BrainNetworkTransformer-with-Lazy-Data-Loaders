from omegaconf import DictConfig, open_dict
from .dataloader import create_stratified_dataloaders
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide']

    #datasets = eval(
    #    f"load_{cfg.dataset.name}_data")(cfg)
    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = (22,22)
        cfg.dataset.timeseries_sz = 2560
    
    dataloaders = create_stratified_dataloaders(cfg, '/home/faizan/MIT/data')

    return dataloaders
