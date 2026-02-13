import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class RecommenderDataset(Dataset):
    """
    Custom PyTorch Dataset for User-Item interactions.
    """
    def __init__(self, users, items, labels):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def get_dataloader(df, batch_size, shuffle=True):
    """
    Creates a DataLoader from a pandas DataFrame.
    DataFrame must have 'user_id', 'item_id', and 'label' columns.
    """
    dataset = RecommenderDataset(
        df['user_id'].values, 
        df['item_id'].values, 
        df['label'].values
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def preprocess_data(df):
    """
    Encodes user and item IDs to continuous integers.
    """
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)
    
    return df, user_map, item_map
