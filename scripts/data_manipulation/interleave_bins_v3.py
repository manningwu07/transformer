#!/usr/bin/env python3
"""
interleave_bins_v3.py - Strict multi-source shard interleaving with DPO support.

Changes from v2:
1. STRICT: RuntimeError if any --source is missing train/ or val/ subdirs
2. RECURSIVE: glob(..., recursive=True) for nested val buckets
3. DPO: Auto-copy dpo/*.npz from sources to output
"""
import argparse
import glob
import os
import random
import shutil
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class ShardWriterFast:
    out_dir: str
    prefix: str
    shard_size: int
    dtype: np.dtype
    shard_idx: int = 0
    total_written: int = 0

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._buf = np.empty((self.shard_size,), dtype=self.dtype)
        self._pos = 0

    def _path(self) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:05d}.bin")

    def _flush_full(self) -> None:
        assert self._pos == self.shard_size
        self._buf.tofile(self._path())
        self.total_written += int(self._pos)
        self.shard_idx += 1
        self._pos = 0

    def push_arr(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        i = 0
        while i < arr.size:
            space = self.shard_size - self._pos
            take = min(space, arr.size - i)
            self._buf[self._pos : self._pos + take] = arr[i : i + take]
            self._pos += take
            i += take
            if self._pos == self.shard_size:
                self._flush_full()

    def finalize(self) -> None:
        if self._pos > 0:
            self._buf[: self._pos].tofile(self._path())
            self.total_written += int(self._pos)
            self.shard_idx += 1
            self._pos = 0


def iter_shard_tokens_recursive(
    shard_dir: str,
    dtype: np.dtype,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """
    Recursively find all .bin files under shard_dir and yield chunks.
    Handles nested structures like val/stackedu/*.bin, val/finemath/*.bin
    """
    # Recursive glob for all .bin files
    pattern = os.path.join(shard_dir, "**", "*.bin")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        raise RuntimeError(f"No .bin files found in {shard_dir} (recursive search)")
    
    for f in files:
        mm = np.memmap(f, dtype=dtype, mode="r")
        n = int(mm.shape[0])
        for start in range(0, n, chunk_size):
            yield np.asarray(mm[start : start + chunk_size])


def interleave_sources(
    sources: List[Iterator[np.ndarray]],
    weights: List[float],
    seed: int,
) -> Iterator[np.ndarray]:
    rng = random.Random(seed)
    active = [True] * len(sources)
    w = [float(x) for x in weights]

    # Prime one chunk per source
    buf: List[np.ndarray | None] = [None] * len(sources)
    for i in range(len(sources)):
        try:
            buf[i] = next(sources[i])
        except StopIteration:
            active[i] = False
            w[i] = 0.0

    while any(active):
        total = sum(w[i] for i in range(len(w)) if active[i] and w[i] > 0.0)
        if total <= 0:
            break

        probs = [w[i] / total if active[i] else 0.0 for i in range(len(w))]
        idx = rng.choices(range(len(sources)), weights=probs, k=1)[0]

        if not active[idx]:
            continue

        # Yield exactly ONE chunk per choice
        if buf[idx] is not None:
            out = buf[idx]
            buf[idx] = None
            yield out
            continue

        try:
            yield next(sources[idx])
        except StopIteration:
            active[idx] = False
            w[idx] = 0.0


def parse_sources(items: List[str]) -> List[Tuple[str, str, float]]:
    """
    Each --source: "name=/path/to/shards:weight"
    Expects /path/to/shards/{train,val}/*.bin (recursive)
    """
    out = []
    for s in items:
        if "=" not in s or ":" not in s:
            raise ValueError(f"Bad --source '{s}'. Use name=/path:weight")
        name, rest = s.split("=", 1)
        path, w = rest.rsplit(":", 1)
        out.append((name.strip(), path.strip(), float(w)))
    if not out:
        raise ValueError("No sources provided.")
    return out


def validate_sources_strict(
    sources: List[Tuple[str, str, float]],
    require_train: bool = True,
    require_val: bool = True,
) -> None:
    """
    STRICT: Raise RuntimeError if any source is missing required splits.
    """
    for name, base, w in sources:
        if not os.path.isdir(base):
            raise RuntimeError(f"Source '{name}' base path does not exist: {base}")
        
        if require_train:
            train_dir = os.path.join(base, "train")
            if not os.path.isdir(train_dir):
                raise RuntimeError(
                    f"Source '{name}' missing train/ subdirectory: {train_dir}\n"
                    f"Use --skip_missing_train to allow skipping (not recommended)"
                )
            # Check for actual .bin files
            bins = glob.glob(os.path.join(train_dir, "**", "*.bin"), recursive=True)
            if not bins:
                raise RuntimeError(
                    f"Source '{name}' train/ has no .bin files: {train_dir}"
                )
        
        if require_val:
            val_dir = os.path.join(base, "val")
            if not os.path.isdir(val_dir):
                raise RuntimeError(
                    f"Source '{name}' missing val/ subdirectory: {val_dir}\n"
                    f"Use --skip_missing_val to allow skipping (not recommended)"
                )
            # Check for actual .bin files (recursive for nested buckets)
            bins = glob.glob(os.path.join(val_dir, "**", "*.bin"), recursive=True)
            if not bins:
                raise RuntimeError(
                    f"Source '{name}' val/ has no .bin files (recursive): {val_dir}"
                )


def copy_dpo_shards(
    sources: List[Tuple[str, str, float]],
    out_dir: str,
) -> int:
    """
    Copy all dpo/*.npz files from sources to out_dir/dpo/
    Returns total files copied.
    """
    dpo_out = os.path.join(out_dir, "dpo")
    copied = 0
    
    for name, base, _ in sources:
        dpo_dir = os.path.join(base, "dpo")
        if not os.path.isdir(dpo_dir):
            continue
        
        npz_files = glob.glob(os.path.join(dpo_dir, "*.npz"))
        if not npz_files:
            continue
        
        os.makedirs(dpo_out, exist_ok=True)
        for f in npz_files:
            # Prefix with source name to avoid collisions
            dst_name = f"{name}_{os.path.basename(f)}"
            dst = os.path.join(dpo_out, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                copied += 1
    
    return copied


def build_split(
    split: str,
    sources: List[Tuple[str, str, float]],
    out_dir: str,
    dtype: np.dtype,
    shard_size: int,
    target_tokens: int,
    seed: int,
    chunk_size: int,
    strict: bool = True,
):
    """Build train or val split from multiple sources."""
    iters: List[Iterator[np.ndarray]] = []
    weights: List[float] = []
    names: List[str] = []

    for name, base, w in sources:
        sd = os.path.join(base, split)
        
        if not os.path.isdir(sd):
            if strict:
                raise RuntimeError(f"STRICT: Missing split dir for '{name}': {sd}")
            print(f"⚠️ missing split dir, skipping: {sd}")
            continue
        
        # Use recursive iterator
        try:
            it = iter_shard_tokens_recursive(sd, dtype, chunk_size)
            iters.append(it)
            weights.append(w)
            names.append(name)
        except RuntimeError as e:
            if strict:
                raise
            print(f"⚠️ {e}")
            continue

    if not iters:
        raise RuntimeError(f"No valid sources for split={split}.")

    out_split = os.path.join(out_dir, split)
    writer = ShardWriterFast(
        out_dir=out_split,
        prefix=f"mixed-{split}",
        shard_size=shard_size,
        dtype=dtype,
    )

    print(f"\n📦 Mixing split={split} -> {out_split}")
    total_w = sum(weights)
    for n, w in zip(names, weights):
        print(f"  - {n}: {w / max(1e-12, total_w):.3f}")

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"mix:{split}")
    for chunk in interleave_sources(iters, weights, seed=seed):
        writer.push_arr(chunk.astype(dtype, copy=False))
        pbar.update(int(chunk.size))
        if writer.total_written >= target_tokens:
            break

    writer.finalize()
    pbar.close()
    print(f"✅ {split}: wrote {writer.total_written:,} tokens ({writer.shard_idx} shards)")
    return writer.total_written


def main():
    ap = argparse.ArgumentParser(
        description="Strict multi-source shard interleaving with DPO support"
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"])
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--chunk_size", type=int, default=262_144)

    ap.add_argument("--target_train_tokens", type=int, required=True)
    ap.add_argument("--target_val_tokens", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help='Repeatable. Format: name=/abs/or/rel/path:weight',
    )
    
    # Strict mode flags (default: strict)
    ap.add_argument(
        "--skip_missing_train",
        action="store_true",
        help="Allow skipping sources with missing train/ (NOT RECOMMENDED)",
    )
    ap.add_argument(
        "--skip_missing_val", 
        action="store_true",
        help="Allow skipping sources with missing val/ (NOT RECOMMENDED)",
    )
    ap.add_argument(
        "--no_dpo",
        action="store_true",
        help="Skip copying DPO shards",
    )

    args = ap.parse_args()
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    sources = parse_sources(args.source)

    # STRICT VALIDATION (before any work)
    print("🔍 Validating sources (STRICT mode)...")
    validate_sources_strict(
        sources,
        require_train=not args.skip_missing_train,
        require_val=not args.skip_missing_val,
    )
    print("✅ All sources validated.\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # Build train split
    build_split(
        split="train",
        sources=sources,
        out_dir=args.out_dir,
        dtype=dtype,
        shard_size=args.shard_size,
        target_tokens=args.target_train_tokens,
        seed=args.seed,
        chunk_size=args.chunk_size,
        strict=not args.skip_missing_train,
    )
    
    # Build val split
    build_split(
        split="val",
        sources=sources,
        out_dir=args.out_dir,
        dtype=dtype,
        shard_size=args.shard_size,
        target_tokens=args.target_val_tokens,
        seed=args.seed + 999,
        chunk_size=args.chunk_size,
        strict=not args.skip_missing_val,
    )
    
    # Copy DPO shards
    if not args.no_dpo:
        print("\n📋 Copying DPO shards...")
        n_dpo = copy_dpo_shards(sources, args.out_dir)
        if n_dpo > 0:
            print(f"✅ Copied {n_dpo} DPO files to {args.out_dir}/dpo/")
        else:
            print("ℹ️ No DPO shards found in sources.")


if __name__ == "__main__":
    main()