#!/usr/bin/env python3
"""
Phase 3 Data Interleaver: Mixes Phase 1 + Phase 2 + Phase 3 (longctx + sft).

For long-context + SFT/DPO training with replay of earlier phases.

Output: data/shards/phase3_mixed/{train,val}/*.bin
"""
import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
from tqdm import tqdm


@dataclass
class ShardMixer:
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
    files = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    if not files:
        return  # Empty iterator for optional sources

    for f in files:
        mm = np.memmap(f, dtype=dtype, mode="r")
        for start in range(0, len(mm), chunk_size):
            yield np.array(mm[start : start + chunk_size])


def interleave_phases(
    sources: List[Iterator[np.ndarray]],
    weights: List[float],
    seed: int,
) -> Iterator[np.ndarray]:
    rng = random.Random(seed)
    active = [True] * len(sources)
    norm_weights = weights[:]
    
    # Prime sources - check which ones actually have data
    buffers = []
    for i, src in enumerate(sources):
        try:
            first = next(src)
            buffers.append(first)
        except StopIteration:
            active[i] = False
            norm_weights[i] = 0.0
            buffers.append(None)

    while any(active):
        total = sum(w for i, w in enumerate(norm_weights) if active[i])
        if total <= 0:
            break

        probs = [w / total if active[i] else 0 for i, w in enumerate(norm_weights)]
        idx = rng.choices(range(len(sources)), weights=probs, k=1)[0]

        if not active[idx]:
            continue

        # Yield buffered chunk
        if buffers[idx] is not None:
            yield buffers[idx]
            buffers[idx] = None

        # Get next chunk
        try:
            chunk = next(sources[idx])
            yield chunk
        except StopIteration:
            active[idx] = False
            norm_weights[idx] = 0.0


def main():
    ap = argparse.ArgumentParser(description="Interleave Phase 1 + 2 + 3 shards")

    ap.add_argument("--phase1_dir", type=str, default=None, help="Phase 1 shard directory (optional)")
    ap.add_argument("--phase2_dir", type=str, default=None, help="Phase 2 shard directory (optional)")
    ap.add_argument("--phase3_longctx_dir", type=str, required=True, help="Phase 3 longctx directory")
    ap.add_argument("--phase3_sft_dir", type=str, required=True, help="Phase 3 SFT directory")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase3_mixed")

    # Weights: Phase3 dominates, with small replay of earlier phases
    ap.add_argument("--phase1_weight", type=float, default=10.0)
    ap.add_argument("--phase2_weight", type=float, default=15.0)
    ap.add_argument("--longctx_weight", type=float, default=50.0)
    ap.add_argument("--sft_weight", type=float, default=25.0)

    ap.add_argument("--target_tokens", type=int, default=25_000_000_000)
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"])

    args = ap.parse_args()

    dtype = np.uint16 if args.dtype == "uint16" else np.uint32

    # Build sources and weights dynamically
    sources = []
    weights = []
    names = []

    # Phase 1 (optional replay)
    if args.phase1_dir and os.path.isdir(args.phase1_dir):
        sources.append(iter_shard_tokens(args.phase1_dir, "*train*.bin", dtype))
        weights.append(args.phase1_weight)
        names.append(f"phase1 ({args.phase1_weight}%)")

    # Phase 2 (optional replay)
    if args.phase2_dir and os.path.isdir(args.phase2_dir):
        sources.append(iter_shard_tokens(args.phase2_dir, "*train*.bin", dtype))
        weights.append(args.phase2_weight)
        names.append(f"phase2 ({args.phase2_weight}%)")

    # Phase 3 longctx (required)
    sources.append(iter_shard_tokens(args.phase3_longctx_dir, "*.bin", dtype))
    weights.append(args.longctx_weight)
    names.append(f"longctx ({args.longctx_weight}%)")

    # Phase 3 SFT (required)
    sources.append(iter_shard_tokens(args.phase3_sft_dir, "*.bin", dtype))
    weights.append(args.sft_weight)
    names.append(f"sft ({args.sft_weight}%)")

    out_train = os.path.join(args.out_dir, "train")
    mixer = ShardMixer(
        out_dir=out_train,
        prefix="phase3-mixed-train",
        shard_size=args.shard_size,
        dtype=dtype,
    )

    print(f"📦 Interleaving for Phase 3:")
    for n in names:
        print(f"   - {n}")
    print(f"   Target: {args.target_tokens:,} tokens")
    print(f"   Output: {out_train}")

    pbar = tqdm(total=args.target_tokens, unit="tok", desc="Mixing")

    for chunk in interleave_phases(sources, weights, args.seed):
        mixer.push(chunk)
        pbar.update(len(chunk))
        if mixer.total_written >= args.target_tokens:
            break

    mixer.finalize()
    pbar.close()

    print(f"✅ Phase 3 mixed complete: {mixer.total_written:,} tokens in {mixer.shard_idx} shards")


if __name__ == "__main__":
    main()