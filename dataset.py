import os
import glob
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import random


class IndexedBinaryDataset(Dataset):
    """
    Reads sharded .bin/.idx pairs from makeDatasetFast.py
    Correctly handles uint32 tokens and variable-length sequences.
    """

    def __init__(self, shard_dir: str, split: str = "train", seq_len: int = 2048):
        self.shard_dir = shard_dir
        self.split = split
        self.seq_len = seq_len

        # Find all shards for this split
        self.bin_files = sorted(glob.glob(os.path.join(shard_dir, f"{split}-*.bin")))
        self.idx_files = sorted(glob.glob(os.path.join(shard_dir, f"{split}-*.idx")))

        if not self.bin_files:
            raise FileNotFoundError(
                f"No shards found at {shard_dir}/{split}-*.bin\n"
                f"Run: python scripts/makeDatasetFast.py"
            )

        # Load all indices into memory (small: 16 bytes per sequence)
        self.sequences = []  # List of (bin_path, byte_offset, token_count)
        for bin_path, idx_path in zip(self.bin_files, self.idx_files):
            with open(idx_path, "rb") as f:
                while True:
                    chunk = f.read(16)  # 2x uint64
                    if not chunk:
                        break
                    byte_offset, token_count = struct.unpack("<QQ", chunk)
                    if token_count >= 4:  # Skip tiny sequences
                        self.sequences.append((bin_path, byte_offset, token_count))

        # Mmap all bin files (lazy, no RAM used until accessed)
        self.mmaps = {}
        for bin_path in self.bin_files:
            self.mmaps[bin_path] = np.memmap(bin_path, dtype=np.uint32, mode="r")

        print(
            f"📂 IndexedBinaryDataset: {len(self.sequences)} sequences "
            f"from {len(self.bin_files)} shards [{split}]"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        bin_path, byte_offset, token_count = self.sequences[idx]

        # Convert byte offset to token offset (uint32 = 4 bytes)
        token_offset = byte_offset // 4

        # Get tokens from mmap
        mmap = self.mmaps[bin_path]
        tokens = mmap[token_offset : token_offset + token_count]

        # Handle sequence length
        if len(tokens) > self.seq_len + 1:
            # Random window (preserves BOS/EOS somewhere in context)
            max_start = len(tokens) - self.seq_len - 1
            start = random.randint(0, max_start)
            tokens = tokens[start : start + self.seq_len + 1]
        elif len(tokens) < self.seq_len + 1:
            # Pad (rare if filtering worked)
            pad_len = self.seq_len + 1 - len(tokens)
            tokens = np.concatenate([tokens, np.zeros(pad_len, dtype=np.uint32)])

        # Convert to tensor
        tokens = torch.from_numpy(tokens.astype(np.int64))

        x = tokens[:-1]  # Input
        y = tokens[1:]  # Target (shifted by 1)

        return x, y


class StreamingShardDataset(IterableDataset):
    """
    Alternative: Streaming dataset for very large corpora.
    Handles multi-worker correctly, lower memory than indexed.
    """

    def __init__(self, shard_dir: str, split: str = "train", seq_len: int = 2048):
        self.shard_dir = shard_dir
        self.split = split
        self.seq_len = seq_len
        self.shards = sorted(glob.glob(os.path.join(shard_dir, f"{split}-*.bin")))

        if not self.shards:
            raise FileNotFoundError(f"No shards found at {shard_dir}/{split}-*.bin")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Split shards across workers
            shards = self.shards[worker_info.id :: worker_info.num_workers]
        else:
            shards = self.shards

        for shard_path in shards:
            data = np.memmap(shard_path, dtype=np.uint32, mode="r")

            # Random offset for this epoch
            if len(data) > self.seq_len + 1:
                offset = random.randint(0, min(1000, len(data) - self.seq_len - 1))
            else:
                offset = 0

            for i in range(offset, len(data) - self.seq_len - 1, self.seq_len):
                chunk = torch.from_numpy(
                    data[i : i + self.seq_len + 1].astype(np.int64)
                )
                yield chunk[:-1], chunk[1:]


# Backwards compatibility alias
BinaryDataset = IndexedBinaryDataset