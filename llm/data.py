import torch
import numpy as np
import os

class DataLoader:
    """
    The DataLoader is responsible for feeding data to the model during training.
    
    It reads a large binary file (containing all our text tokens) efficiently 
    and serves up random chunks (batches) for the model to learn from.
    """

    def __init__(self, data_path: str, block_size: int, batch_size: int, split: str = 'train'):
        """
        Initialize the DataLoader.

        Args:
            data_path (str): The folder where the .bin files are stored.
            block_size (int): Also called 'context length'. How many tokens the model sees at once.
                              (e.g., if block_size is 8, the model looks at 8 words to predict the 9th).
            batch_size (int): How many independent sequences to process in parallel.
                              (e.g., if batch_size is 4, we feed 4 different sentences at the same time).
            split (str): Which file to load? 'train' for training data, 'val' for validation data.
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split
        
        # Construct the full path to the file, e.g., "data/train.bin"
        filename = os.path.join(data_path, f'{split}.bin')
        
        # Safety check: Ensure the data file actually exists before crashing later
        if not os.path.exists(filename):
             raise FileNotFoundError(f"Data file {filename} not found. Run data/prepare.py first.")
             
        # KEY CONCEPT: np.memmap (Memory Map)
        # Instead of reading the whole 10GB file into RAM (which would crash your computer),
        # memmap creates a 'virtual' link to the file on the hard drive.
        # It allows us to access parts of the file as if it were in memory, 
        # but the OS handles loading only the tiny bits we actually need at that moment.
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        self.data_len = len(self.data)
        
    def get_batch(self):
        """
        Fetches a random batch of data for training.
        
        Returns:
            x (torch.Tensor): The input sequences (shape: batch_size x block_size).
            y (torch.Tensor): The target sequences (shape: batch_size x block_size).
            
        HOW THE TARGETS WORK:
        We are training a Language Model to predict the NEXT token.
        If the data is: [10, 20, 30, 40, 50]
        And block_size is 4:
        
        Input (x):  [10, 20, 30, 40]  (The context)
        Target (y): [20, 30, 40, 50]  (The correct next tokens)
        
        Notice 'y' is just 'x' shifted to the right by 1 position.
        """
        # 1. Generate random starting positions (indices) in the data.
        # We subtract block_size to ensure we don't go off the end of the file.
        # ix will look like: [4502, 120, 99532, 12] (random spots in the book)
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        
        # 2. Grab the INPUT chunks (x)
        # For each random spot 'i', grab 'block_size' amount of tokens.
        # We cast to int64 because PyTorch usually expects Long/Int64 for indices.
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        
        # 3. Grab the TARGET chunks (y)
        # This is exactly the same as x, but shifted by +1 (i+1).
        # We want the model to predict the token at i+1 given the token at i.
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        # Return the tensors to be fed into the GPU/Model
        return x, y

    def __len__(self):
        """
        Returns the approximate number of batches in the dataset.
        Useful for progress bars (tqdm) to know how long an 'epoch' is.
        """
        return self.data_len // (self.block_size * self.batch_size)