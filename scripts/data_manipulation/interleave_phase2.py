#!/usr/bin/env python3
"""
Phase 2 Data Interleaver: Mixes Phase 1 (pretrain) + Phase 2 (reasoning) shards.

Prevents catastrophic forgetting by replaying base pretraining data
while introducing new reasoning/math/code data.

Warmup: 30% Phase 1, 70% Phase 2
Steady-state: 10% Phase 1, 90% Phase 2

Output: data/shards/phase2_{warmup/final}/{train,val}/*.bin
"""
import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
from tqdm import tqdm


@dataclass
class ShardMixer:
    """Interleaves tokens from multiple phase directories by probability."""

    out_dir: str
    prefix: str
    shard_size: int
    dtype: np.dtype
    shard_idx: int = 0
    total_written: int = 0
    buf: List[int] = None

    def __post_init__(self):
        self.buf = []
        os.makedirs(self.out_dir, exist_ok=True)

    def _path(self) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:05d}.bin")

    def push(self, tokens: np.ndarray) -> None:
        self.buf.extend(tokens.tolist())
        self._flush()

    def _flush(self) -> None:
        while len(self.buf) >= self.shard_size:
            chunk = self.buf[: self.shard_size]
            arr = np.asarray(chunk, dtype=self.dtype)
            arr.tofile(self._path())
            self.total_written += arr.size
            self.shard_idx += 1
            self.buf = self.buf[self.shard_size :]

    def finalize(self) -> None:
        if self.buf:
            arr = np.asarray(self.buf, dtype=self.dtype)
            arr.tofile(self._path())
            self.total_written += arr.size
            self.shard_idx += 1
            self.buf = []


def iter_shard_tokens(
    shard_dir: str,
    pattern: str,
    dtype: np.dtype,
    chunk_size: int = 100_000,
) -> Iterator[np.ndarray]:
    """Yield chunks of tokens from all shards in a directory."""
    files = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No shards matching {pattern} in {shard_dir}")

    for f in files:
        mm = np.memmap(f, dtype=dtype, mode="r")
        for start in range(0, len(mm), chunk_size):
            yield np.array(mm[start : start + chunk_size])


def interleave_phases(
    sources: List[Iterator[np.ndarray]],
    weights: List[float],
    seed: int,
) -> Iterator[np.ndarray]:
    """
    Probabilistically interleave multiple token streams.
    Removes exhausted sources dynamically.
    """
    rng = random.Random(seed)
    active = [True] * len(sources)
    norm_weights = weights[:]

    while any(active):
        # Renormalize weights for active sources
        total = sum(w for i, w in enumerate(norm_weights) if active[i])
        if total <= 0:
            break

        probs = [w / total if active[i] else 0 for i, w in enumerate(norm_weights)]
        idx = rng.choices(range(len(sources)), weights=probs, k=1)[0]

        if not active[idx]:
            continue

        try:
            chunk = next(sources[idx])
            yield chunk
        except StopIteration:
            active[idx] = False
            norm_weights[idx] = 0.0


def main():
    ap = argparse.ArgumentParser(description="Interleave Phase 1 + Phase 2 shards")

    ap.add_argument("--phase1_dir", type=str, required=True, help="Phase 1 shard directory")
    ap.add_argument("--phase2_dir", type=str, required=True, help="Phase 2 shard directory")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase2_mixed")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])

    # Mixing weights (default: 30% phase1 replay, 70% phase2 new)
    ap.add_argument("--phase1_weight", type=float, default=30.0)
    ap.add_argument("--phase2_weight", type=float, default=70.0)

    ap.add_argument("--target_tokens", type=int, default=50_000_000_000, help="Target token count")
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"])

    args = ap.parse_args()

    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    pattern = f"*{args.split}*.bin"

    # Validate directories
    for d, name in [(args.phase1_dir, "phase1"), (args.phase2_dir, "phase2")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{name}_dir not found: {d}")
        if not glob.glob(os.path.join(d, pattern)):
            raise FileNotFoundError(f"No {args.split} shards in {d}")

    out_subdir = os.path.join(args.out_dir, args.split)
    mixer = ShardMixer(
        out_dir=out_subdir,
        prefix=f"phase2-mixed-{args.split}",
        shard_size=args.shard_size,
        dtype=dtype,
    )

    # Create iterators
    sources = [
        iter_shard_tokens(args.phase1_dir, pattern, dtype),
        iter_shard_tokens(args.phase2_dir, pattern, dtype),
    ]
    weights = [args.phase1_weight, args.phase2_weight]

    print(f"📦 Interleaving Phase 1 ({args.phase1_weight}%) + Phase 2 ({args.phase2_weight}%)")
    print(f"   Split: {args.split}")
    print(f"   Target: {args.target_tokens:,} tokens")
    print(f"   Output: {out_subdir}")

    pbar = tqdm(total=args.target_tokens, unit="tok", desc="Mixing")

    for chunk in interleave_phases(sources, weights, args.seed):
        mixer.push(chunk)
        pbar.update(len(chunk))
        if mixer.total_written >= args.target_tokens:
            break

    mixer.finalize()
    pbar.close()

    print(f"✅ Phase 2 mixed complete: {mixer.total_written:,} tokens in {mixer.shard_idx} shards")


if __name__ == "__main__":
    main()