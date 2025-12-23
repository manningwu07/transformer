import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

class PackedBinDataset(Dataset):
    """
    Reads *flat* packed token stream shards: just .bin files, no .idx.
    Each shard is a contiguous array of token IDs (uint16 or uint32).
    We sample fixed windows of length (seq_len + 1) to form (x, y).
    """

    def __init__(
        self,
        shard_dir: str,
        split: str = "train",
        seq_len: int = 2048,
        dtype: np.dtype = np.uint16,
        pattern: str | None = None,
    ):
        self.shard_dir = shard_dir
        self.split = split
        self.seq_len = seq_len
        self.block = seq_len + 1
        self.dtype = dtype

        # Your files are named like: phase1-train-000.bin
        # Default pattern: anything containing the split string.
        pat = pattern or f"*{split}*.bin"
        self.bin_files = sorted(glob.glob(os.path.join(shard_dir, pat)))

        if not self.bin_files:
            raise FileNotFoundError(
                f"No .bin shards found in {shard_dir} matching pattern '{pat}'.\n"
                f"Examples expected: phase1-train-000.bin"
            )

        # Memmap shards + build cumulative token offsets for global addressing
        self.mmaps: list[np.memmap] = []
        self.cum_tokens = [0]  # cum_tokens[i] = total tokens before shard i

        total = 0
        for p in self.bin_files:
            m = np.memmap(p, dtype=self.dtype, mode="r")
            self.mmaps.append(m)
            total += int(m.shape[0])
            self.cum_tokens.append(total)

        self.total_tokens = total
        self.num_blocks = self.total_tokens // self.block

        # Drop last partial block (no padding)
        if self.num_blocks <= 0:
            raise ValueError(
                f"Not enough tokens ({self.total_tokens}) for seq_len={seq_len}."
            )

        print(
            f"📦 PackedBinDataset: {self.total_tokens:,} tokens "
            f"across {len(self.bin_files)} shards | "
            f"{self.num_blocks:,} blocks of {self.block} tokens"
        )

    def __len__(self):
        return self.num_blocks

    def _slice_global(self, start: int, length: int) -> np.ndarray:
        """
        Return a contiguous slice [start:start+length] across shard boundaries.
        """
        end = start + length
        if end > self.total_tokens:
            raise IndexError("Requested slice exceeds total_tokens")

        out = []
        cur = start

        while cur < end:
            shard_idx = bisect.bisect_right(self.cum_tokens, cur) - 1
            shard_start = self.cum_tokens[shard_idx]
            local = cur - shard_start

            shard = self.mmaps[shard_idx]
            take = min(end - cur, shard.shape[0] - local)
            out.append(np.asarray(shard[local : local + take]))
            cur += take

        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=0)

    def __getitem__(self, idx: int):
        # Deterministic block start; DataLoader(shuffle=True) randomizes block order
        start = idx * self.block
        tokens = self._slice_global(start, self.block).astype(np.int64, copy=False)

        x = torch.from_numpy(tokens[:-1])
        y = torch.from_numpy(tokens[1:])
        return x, y

# For backward compatibility
BinaryDataset = PackedBinDataset