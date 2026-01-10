#!/usr/bin/env python3
import argparse
import glob
import os
import random
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


def iter_shard_tokens(
    shard_dir: str,
    pattern: str,
    dtype: np.dtype,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    files = sorted(glob.glob(os.path.join(shard_dir, pattern)))
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
    Expects /path/to/shards/{train,val}/*.bin
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


def build_split(
    split: str,
    sources: List[Tuple[str, str, float]],
    out_dir: str,
    dtype: np.dtype,
    shard_size: int,
    target_tokens: int,
    seed: int,
    chunk_size: int,
):
    iters: List[Iterator[np.ndarray]] = []
    weights: List[float] = []
    names: List[str] = []

    for name, base, w in sources:
        sd = os.path.join(base, split)
        if not os.path.isdir(sd):
            print(f"⚠️ missing split dir, skipping: {sd}")
            continue
        iters.append(iter_shard_tokens(sd, "*.bin", dtype, chunk_size))
        weights.append(w)
        names.append(name)

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


def main():
    ap = argparse.ArgumentParser()
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

    args = ap.parse_args()
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    sources = parse_sources(args.source)

    os.makedirs(args.out_dir, exist_ok=True)

    build_split(
        split="train",
        sources=sources,
        out_dir=args.out_dir,
        dtype=dtype,
        shard_size=args.shard_size,
        target_tokens=args.target_train_tokens,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    build_split(
        split="val",
        sources=sources,
        out_dir=args.out_dir,
        dtype=dtype,
        shard_size=args.shard_size,
        target_tokens=args.target_val_tokens,
        seed=args.seed + 999,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()