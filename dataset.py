import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
import bisect

class PackedBinDataset(Dataset):
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

        pat = pattern or f"*{split}*.bin"
        self.bin_files = sorted(glob.glob(os.path.join(shard_dir, pat)))
        if not self.bin_files:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

        self.mmaps = []
        self.cum_tokens = [0]
        total = 0
        for p in self.bin_files:
            m = np.memmap(p, dtype=self.dtype, mode="r")
            self.mmaps.append(m)
            total += int(m.shape[0])
            self.cum_tokens.append(total)

        self.total_tokens = total
        self.num_blocks = self.total_tokens // self.block

    def __len__(self):
        return self.num_blocks

    def _slice_global(self, start: int, length: int) -> np.ndarray:
        end = start + length
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
        return out[0] if len(out) == 1 else np.concatenate(out, axis=0)

    def __getitem__(self, idx: int):
        start = idx * self.block
        tokens = self._slice_global(start, self.block)
        return torch.from_numpy(tokens[:-1]), torch.from_numpy(tokens[1:])

class ResumableSampler(Sampler):
    def __init__(self, sampler, start_idx):
        self.sampler = sampler
        self.start_idx = start_idx

    def __iter__(self):
        indices = list(self.sampler)
        return iter(indices[self.start_idx :])

    def __len__(self):
        return max(0, len(self.sampler) - self.start_idx)

def get_dataloader(
    ds,
    sampler,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    seed: int,
    offset: int = 0,
    drop_last: bool = True,
):
    # If a sampler is provided, DataLoader requires shuffle=False
    if sampler is not None:
        shuffle = False

    # Interpret offset as "number of batches already consumed" (your train.py passes
    # micro_step_in_epoch), convert to "number of sample indices to skip".
    if offset > 0:
        base_sampler = sampler or RandomSampler(ds)
        start_idx = int(offset) * int(batch_size)
        sampler = ResumableSampler(base_sampler, start_idx)
        shuffle = False

    g = torch.Generator()
    g.manual_seed(int(seed))

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=(int(num_workers) > 0),
        prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
        drop_last=bool(drop_last),
        generator=g,
    )
