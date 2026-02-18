import torch
import numpy as np
import os

class DataLoader:
    def __init__(self, data_path: str, block_size: int, batch_size: int, split: str = 'train'):
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split
        
        filename = os.path.join(data_path, f'{split}.bin')
        if not os.path.exists(filename):
             raise FileNotFoundError(f"Data file {filename} not found. Run data/prepare.py first.")
             
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        self.data_len = len(self.data)
        
    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        return x, y

    def __len__(self):
        return self.data_len // (self.block_size * self.batch_size)
