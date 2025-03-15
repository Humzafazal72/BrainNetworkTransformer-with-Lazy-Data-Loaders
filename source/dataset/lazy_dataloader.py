import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict

class StratifiedTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to directory with two class folders
            transform (callable, optional): Optional transform to be applied
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Find all classes (subdirectories)
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Collect all file paths and labels
        self.data_paths = []
        self.labels = []
        
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            class_files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                           if os.path.isfile(os.path.join(class_path, f))]
            
            self.data_paths.extend(class_files)
            self.labels.extend([i] * len(class_files))
        
        # Convert to numpy arrays for stratified splitting
        self.data_paths = np.array(self.data_paths)
        self.labels = np.array(self.labels)

    def calculate_safe_correlation(self,data):
        eps = 1e-8

        mean = np.mean(data, axis=1, keepdims=True)  # Use keepdims=True
        std = np.std(data, axis=1, keepdims=True)    # Use keepdims=True

        std = np.where(std < eps, eps, std)

        normalized_data = (data - mean) / std

        corr = np.dot(normalized_data, normalized_data.T) / (data.shape[1] - 1)

        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 0.0)

        return corr

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
    # Load data (customize based on your specific data format)
        data_path = self.data_paths[idx]
        label = self.labels[idx]

        # Load and transform data
        timeseires = np.load(data_path, allow_pickle=True)['data'][0:22]
        pearson = self.calculate_safe_correlation(timeseires)

        # Convert data to tensors
        final_timeseires = torch.from_numpy(timeseires).float()
        final_pearson = torch.from_numpy(pearson).float()

        # One-hot encode the label
        final_label = torch.tensor(label, dtype=torch.int64)  # Convert label to tensor
        final_label = F.one_hot(final_label, num_classes=len(self.classes))  # Ensure one-hot encoding
        final_label = final_label  # Add batch dimension to make shape consistent

        #print(f"Timeseires shape: {final_timeseires.shape}")
        #print(f"Pearson shape: {final_pearson.shape}")
        #print(f"Label shape: {final_label.shape}")

        return final_timeseires, final_pearson, final_label


def create_stratified_lazy_dataloaders(cfg, data_dir):
    """
    Create stratified train/val/test dataloaders
    
    Args:
        cfg (DictConfig): Configuration object
        data_dir (str): Path to dataset directory
    
    Returns:
        List of DataLoaders [train, val, test]
    """
    # Create full dataset
    full_dataset = StratifiedTimeSeriesDataset(data_dir)
    
    # Stratified splitting
    length = len(full_dataset)
    train_length = int(length * cfg.dataset.train_set * cfg.datasz.percentage)
    val_length = int(length * cfg.dataset.val_set)
    test_length = length - train_length - val_length

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
    
    # Perform stratified split
    split = StratifiedShuffleSplit(n_splits=1, 
                                   train_size=train_length, 
                                   test_size=val_length+test_length, 
                                   random_state=42)
    
    train_idx, test_val_idx = next(split.split(full_dataset.data_paths, full_dataset.labels))
    
    # Further split test/val
    val_test_split = StratifiedShuffleSplit(n_splits=1, 
                                            test_size=test_length)
    test_idx, val_idx = next(val_test_split.split(
        full_dataset.data_paths[test_val_idx], 
        full_dataset.labels[test_val_idx]
    ))
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, test_val_idx[val_idx])
    test_dataset = torch.utils.data.Subset(full_dataset, test_val_idx[test_idx])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        drop_last=cfg.dataset.drop_last
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    return [train_loader, val_loader, test_loader]