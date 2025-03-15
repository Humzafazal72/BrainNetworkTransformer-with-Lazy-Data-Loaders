from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .dataloader import init_dataloader, init_stratified_dataloader
from .lazy_dataloader import create_stratified_lazy_dataloaders
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide', 'custom']

    if cfg.dataset.name == 'custom':
        
        node_sz = int(input("Enter the total number of nodes: "))
        node_feature_sz = int(input("Enter the node features size: "))
        timeseries_sz = int(input("Enter the raw timeseries size: "))

        with open_dict(cfg):
            cfg.dataset.node_sz, cfg.dataset.node_feature_sz = (node_sz,node_feature_sz)
            cfg.dataset.timeseries_sz = timeseries_sz
    
        dataloaders = create_stratified_lazy_dataloaders(cfg, cfg.dataset.path)

        return dataloaders

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    return dataloaders