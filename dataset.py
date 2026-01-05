import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
import bisect
import itertools
import math

from utils import get_rank, is_distributed

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
        return torch.from_numpy(tokens[:-1].copy()), torch.from_numpy(tokens[1:].copy())

class WeightedPhaseDataset(Dataset):
    """
    Mixes multiple PackedBinDatasets by fixed probabilities.

    - Deterministic per (idx, seed, rank) so it plays nice with resuming.
    - Length is the max length of the component datasets (so samplers work).
    - Sampling is "with replacement" via modulo / randint.
    """

    def __init__(
        self,
        phases: list[tuple[str, Dataset]],
        weights: list[float],
        seed: int = 1337,
    ):
        if len(phases) == 0:
            raise ValueError("WeightedPhaseDataset: no phases provided")
        if len(phases) != len(weights):
            raise ValueError("WeightedPhaseDataset: phases and weights mismatch")

        wsum = float(sum(weights))
        if wsum <= 0:
            raise ValueError("WeightedPhaseDataset: weights sum must be > 0")

        self.phases = phases
        self.names = [n for (n, _) in phases]
        self.datasets = [d for (_, d) in phases]
        self.weights = [float(w) / wsum for w in weights]
        self.seed = int(seed)
        self.rank = int(get_rank())

        # Precompute CDF for sampling
        cdf = []
        acc = 0.0
        for w in self.weights:
            acc += w
            cdf.append(acc)
        cdf[-1] = 1.0
        self.cdf = cdf

        self._len = max(len(d) for d in self.datasets)

    def __len__(self) -> int:
        return self._len

    def _pick_phase(self, idx: int) -> int:
        # Mix in rank so each process does not perfectly mirror the same phase picks
        rng = random.Random(self.seed + 1_000_000 * self.rank + idx)
        r = rng.random()
        for i, p in enumerate(self.cdf):
            if r < p:
                return i
        return len(self.cdf) - 1

    def __getitem__(self, idx: int):
        pi = self._pick_phase(idx)
        ds = self.datasets[pi]

        # Map idx into that dataset deterministically.
        # Using modulo avoids extra RNG calls and is resume-friendly.
        j = idx % len(ds)
        return ds[j]


def _validate_pct_sum(pcts: list[tuple[str, float]]) -> list[tuple[str, float]]:
    pcts = [(k, float(v)) for (k, v) in pcts if float(v) > 0.0]
    if not pcts:
        raise ValueError("No phases enabled (all percentages are 0).")
    s = sum(v for _, v in pcts)
    # Be strict: user asked "make sure adds up to 100 percent"
    if abs(s - 100.0) > 1e-6:
        raise ValueError(f"Phase percentages must sum to 100. Got {s}.")
    return pcts


def build_phase_mix_dataset_from_args(
    args,
    split: str,
    seq_len: int,
    dtype: np.dtype,
    seed: int,
    pattern: str | None = None,
) -> Dataset:
    """
    Expects args to have:
      - args.phase1_dir / args.phase2_dir / args.phase3_dir (optional)
      - args.phase1_pct / args.phase2_pct / args.phase3_pct (floats/ints)

    Returns:
      - PackedBinDataset if only one phase enabled
      - WeightedPhaseDataset if multiple phases enabled
    """

    pcts = _validate_pct_sum(
        [
            ("phase1", getattr(args, "phase1_pct", 0.0)),
            ("phase2", getattr(args, "phase2_pct", 0.0)),
            ("phase3", getattr(args, "phase3_pct", 0.0)),
        ]
    )

    phase_to_dir = {
        "phase1": getattr(args, "phase1_dir", None),
        "phase2": getattr(args, "phase2_dir", None),
        "phase3": getattr(args, "phase3_dir", None),
    }

    phases: list[tuple[str, Dataset]] = []
    weights: list[float] = []
    for name, pct in pcts:
        shard_dir = phase_to_dir.get(name, None)
        if not shard_dir:
            raise ValueError(
                f"{name}_pct={pct} but {name}_dir is not set. "
                f"Provide --{name}_dir."
            )
        ds = PackedBinDataset(
            shard_dir=shard_dir,
            split=split,
            seq_len=seq_len,
            dtype=dtype,
            pattern=pattern,
        )
        phases.append((name, ds))
        weights.append(pct)

    if len(phases) == 1:
        return phases[0][1]

    return WeightedPhaseDataset(phases=phases, weights=weights, seed=seed)

class StreamingRandomSampler(Sampler[int]):
    """
    Memory-safe sampler: yields random indices lazily (with replacement),
    avoids building randperm(n) in RAM.
    Deterministic across runs given (seed, epoch, rank).
    """
    def __init__(self, ds: Dataset, seed: int, epoch: int = 0):
        self.ds = ds
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.rank = int(get_rank())
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        n = len(ds)
        # Give each rank roughly 1/world_size of the "epoch length"
        self.num_samples = int(math.ceil(n / max(1, self.world_size)))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = len(self.ds)
        g = torch.Generator()
        # Mix in rank so each process gets a different stream
        g.manual_seed(self.seed + 1_000_000 * self.epoch + 10_000 * self.rank)
        for _ in range(self.num_samples):
            yield int(torch.randint(0, n, (1,), generator=g).item())

class ResumableSampler(Sampler):
    def __init__(self, sampler, start_idx):
        self.sampler = sampler
        self.start_idx = start_idx

    def __iter__(self):
        return itertools.islice(iter(self.sampler), int(self.start_idx), None)

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
    worker_init_fn=None,
):
    # If a sampler is provided, DataLoader requires shuffle=False
    if sampler is None and shuffle:
        sampler = StreamingRandomSampler(ds, seed=int(seed), epoch=0)
        shuffle = False

    # Interpret offset as "number of batches already consumed" (your train.py passes
    # micro_step_in_epoch), convert to "number of sample indices to skip".
    if offset > 0:
        base_sampler = sampler or StreamingRandomSampler(ds, seed=int(seed), epoch=0)
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
        worker_init_fn=worker_init_fn,
    )
