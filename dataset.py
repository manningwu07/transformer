import torch
import numpy as np
from torch.utils.data import Dataset
import os

class BinaryDataset(Dataset):
    """
    Zero-Copy Memory Mapped Dataset.
    Reads directly from a binary file of uint16 tokens.
    """
    def __init__(self, data_dir, seq_len):
        self.seq_len = seq_len
        
        # Look for train.bin or val.bin
        bin_path = os.path.join(data_dir, "data.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"❌ Could not find {bin_path}. Run prepare_data.py first!")
            
        # Get file size to determine number of tokens
        file_size = os.path.getsize(bin_path)
        # uint16 = 2 bytes per token
        self.total_tokens = file_size // 2
        
        # Memory map the file (Zero RAM usage, OS handles caching)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        # Calculate total batches available
        # We need seq_len + 1 tokens (Input + Target)
        self.num_samples = (self.total_tokens - 1) // seq_len
        print(f"📂 Loaded {bin_path}: {self.num_samples} sequences ({self.total_tokens/1e9:.2f}B tokens)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Calculate start position
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        # Slice from memory map (Instant)
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        # x = input, y = target (shifted by 1)
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y