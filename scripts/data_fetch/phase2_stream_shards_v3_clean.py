#!/usr/bin/env python3
"""
Phase 2 Sharder v3 (CLEAN train/val)

Key properties:
- Single-pass stream per dataset (we always stream HF split="train")
- Deterministic hash split per example text -> disjoint train vs val
- Train is a weighted mix (finemath/stackedu/anchor)
- Val is written into per-bucket dirs (finemath/stackedu/anchor)
- Saturating targets: if val is full, route to train; if train is full, route to val

Outputs:
  out_dir/train/phase2-train-mixed-*.bin
  out_dir/val/finemath/phase2-val-finemath-*.bin
  out_dir/val/stackedu/phase2-val-stackedu-*.bin
  out_dir/val/anchor/phase2-val-anchor-*.bin   (if enabled)
"""

import argparse
import hashlib
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional, List, Tuple, Dict

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


# ----------------------------
# Fast writer (numpy buffer)
# ----------------------------
@dataclass
class BinShardWriterFast:
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

    def push_ids(self, ids: List[int]) -> None:
        if not ids:
            return
        arr = np.asarray(ids, dtype=self.dtype)
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


# ----------------------------
# Prefetch (HF streaming can stall)
# ----------------------------
class PrefetchIter:
    def __init__(self, name: str, it: Iterator[Tuple[str, str]], max_buffer: int = 64):
        self.name = name
        self._it = it
        self._q: queue.Queue[Optional[Tuple[str, str]]] = queue.Queue(maxsize=max_buffer)
        self._done = threading.Event()
        self._err: Optional[BaseException] = None
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        try:
            for item in self._it:
                self._q.put(item)
            self._q.put(None)
        except BaseException as e:
            self._err = e
            try:
                self._q.put(None)
            except Exception:
                pass
        finally:
            self._done.set()

    def try_get(self) -> Optional[Tuple[str, str]]:
        if self._err is not None:
            e = self._err
            self._err = None
            raise e
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def alive(self) -> bool:
        return (not self._done.is_set()) or (not self._q.empty())


def interleave_nonblocking(
    sources: List[PrefetchIter],
    probs: List[float],
    seed: int,
    idle_sleep_sec: float = 0.01,
) -> Iterator[Tuple[str, str]]:
    rng = random.Random(seed)
    while any(s.alive() for s in sources):
        available: list[int] = []
        weights: list[float] = []
        for i, s in enumerate(sources):
            if probs[i] <= 0.0:
                continue
            if s._q.empty():
                continue
            available.append(i)
            weights.append(probs[i])

        if not available:
            time.sleep(idle_sleep_sec)
            continue

        idx = rng.choices(available, weights=weights, k=1)[0]
        item = sources[idx].try_get()
        if item is None:
            continue
        bucket, text = item
        text = (text or "").strip()
        if text:
            yield bucket, text


# ----------------------------
# HF loading
# ----------------------------
def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )


def load_dataset_streaming(
    name: str,
    config: Optional[str],
    token: Optional[str],
):
    # Always stream "train" once; we do hash split ourselves
    kwargs = {"split": "train", "streaming": True}
    args = (name, config) if config is not None else (name,)

    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token

    if token:
        try:
            ds = load_dataset(*args, token=token, **kwargs)
        except TypeError:
            ds = load_dataset(*args, use_auth_token=token, **kwargs)
    else:
        ds = load_dataset(*args, **kwargs)

    return ds.with_format("python")


# ----------------------------
# Split + dedup helpers
# ----------------------------
def assign_split(text: str, val_percent: float) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, "little") % 10_000
    return "val" if x < int(val_percent * 100) else "train"


class RollingDedup:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._seen: set[str] = set()
        self._n = 0

    def keep(self, text: str) -> bool:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        if h in self._seen:
            return False
        self._seen.add(h)
        self._n += 1
        if self._n >= self.capacity:
            self._seen = set()
            self._n = 0
        return True


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


# ----------------------------
# Source iterators (yield (bucket, text))
# ----------------------------
def finemath_texts(hf_token: Optional[str], config: str) -> Iterator[Tuple[str, str]]:
    ds = load_dataset_streaming("HuggingFaceTB/finemath", config=config, token=hf_token)
    for ex in ds:
        t = _safe_str(ex.get("text"))
        if t:
            yield "finemath", t


def stackedu_texts(hf_token: Optional[str], config: str) -> Iterator[Tuple[str, str]]:
    ds = load_dataset_streaming("HuggingFaceTB/stack-edu", config=config, token=hf_token)
    keys = ["content", "text", "code"]
    for ex in ds:
        t = ""
        for k in keys:
            t = _safe_str(ex.get(k))
            if t:
                break
        if t:
            yield "stackedu", t


def anchor_texts(hf_token: Optional[str], name: str, config: str) -> Iterator[Tuple[str, str]]:
    ds = load_dataset_streaming(name, config=config, token=hf_token)
    for ex in ds:
        t = _safe_str(ex.get("text"))
        if t:
            yield "anchor", t


# ----------------------------
# Main sharding loop (single mixed stream -> train + per-bucket val)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase2_v3_clean")
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_percent", type=float, default=2.0)

    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)
    ap.add_argument("--target_train_tokens", type=int, default=4_500_000_000)
    ap.add_argument(
        "--target_val_tokens_per_bucket",
        type=int,
        default=50_000_000,
        help="Each enabled bucket gets this many val tokens.",
    )

    ap.add_argument("--prefetch_buffer", type=int, default=64)
    ap.add_argument("--encode_batch_size", type=int, default=512)

    ap.add_argument("--dedup_capacity", type=int, default=300_000)

    # sources
    ap.add_argument("--finemath_config", type=str, default="finemath-4plus")
    ap.add_argument("--stack_edu_configs", type=str, default="TypeScript,Go,Cpp,Rust")

    ap.add_argument("--enable_anchor", action="store_true")
    ap.add_argument("--anchor_name", type=str, default="HuggingFaceTB/smollm-corpus")
    ap.add_argument("--anchor_config", type=str, default="fineweb-edu-dedup")

    # weights for TRAIN mixing
    ap.add_argument(
        "--train_probs",
        type=str,
        default="0.60,0.40",
        help="If no anchor: finemath,stackedu. If anchor: finemath,stackedu,anchor.",
    )

    args = ap.parse_args()
    hf_token = _hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")

    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    stack_cfgs = [c.strip() for c in args.stack_edu_configs.split(",") if c.strip()]
    if any(c.lower() == "python" for c in stack_cfgs):
        raise ValueError("Do not include Python in --stack_edu_configs for Phase2.")

    probs = [float(x) for x in args.train_probs.split(",") if x.strip()]
    want = 3 if args.enable_anchor else 2
    if len(probs) != want:
        raise ValueError(f"--train_probs must have {want} values.")
    s = sum(probs)
    probs = [p / s for p in probs]

    # Writers
    train_out = BinShardWriterFast(
        out_dir=os.path.join(args.out_dir, "train"),
        prefix="phase2-train-mixed",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )
    val_writers: Dict[str, BinShardWriterFast] = {
        "finemath": BinShardWriterFast(
            out_dir=os.path.join(args.out_dir, "val", "finemath"),
            prefix="phase2-val-finemath",
            shard_size=args.bin_shard_size,
            dtype=dtype,
        ),
        "stackedu": BinShardWriterFast(
            out_dir=os.path.join(args.out_dir, "val", "stackedu"),
            prefix="phase2-val-stackedu",
            shard_size=args.bin_shard_size,
            dtype=dtype,
        ),
    }
    if args.enable_anchor:
        val_writers["anchor"] = BinShardWriterFast(
            out_dir=os.path.join(args.out_dir, "val", "anchor"),
            prefix="phase2-val-anchor",
            shard_size=args.bin_shard_size,
            dtype=dtype,
        )

    # Counters
    train_written = 0
    val_written: Dict[str, int] = {k: 0 for k in val_writers.keys()}
    val_target = int(args.target_val_tokens_per_bucket)

    def all_val_done() -> bool:
        return all(val_written[k] >= val_target for k in val_writers.keys())

    dedup = RollingDedup(args.dedup_capacity)

    # Build sources for mixing
    sources: List[Iterator[Tuple[str, str]]] = []
    weights: List[float] = []

    # FineMath is one source
    sources.append(finemath_texts(hf_token, args.finemath_config))
    weights.append(probs[0])

    # Stack-Edu configs: split stack weight across configs
    stack_weight = probs[1]
    per_cfg = stack_weight / max(1, len(stack_cfgs))
    for cfg in stack_cfgs:
        sources.append(stackedu_texts(hf_token, cfg))
        weights.append(per_cfg)

    # Anchor optional (one source)
    if args.enable_anchor:
        sources.append(anchor_texts(hf_token, args.anchor_name, args.anchor_config))
        weights.append(probs[2])

    # Normalize weights
    total_w = sum(weights)
    weights = [w / max(1e-12, total_w) for w in weights]

    prefetch = [
        PrefetchIter(name=f"src_{i}", it=it, max_buffer=args.prefetch_buffer)
        for i, it in enumerate(sources)
    ]
    mixed = interleave_nonblocking(prefetch, probs=weights, seed=args.seed)

    pbar_train = tqdm(total=args.target_train_tokens, unit="tok", desc="phase2 (train)")
    pbar_val = {
        k: tqdm(total=val_target, unit="tok", desc=f"phase2 (val:{k})")
        for k in val_writers.keys()
    }

    train_batch: List[str] = []
    val_batch: Dict[str, List[str]] = {k: [] for k in val_writers.keys()}

    def flush_text_batch(texts: List[str]) -> List[List[int]]:
        encs = tok.encode_batch(texts)
        out_ids = []
        for enc in encs:
            ids = enc.ids
            if ids and ids[-1] != eos_id:
                ids.append(eos_id)
            if ids:
                out_ids.append(ids)
        return out_ids

    def push_ids_to_train(ids: List[int]) -> None:
        nonlocal train_written
        train_out.push_ids(ids)
        train_written += len(ids)
        pbar_train.update(len(ids))

    def push_ids_to_val(bucket: str, ids: List[int]) -> None:
        val_writers[bucket].push_ids(ids)
        val_written[bucket] += len(ids)
        pbar_val[bucket].update(len(ids))

    while train_written < args.target_train_tokens or not all_val_done():
        bucket, text = next(mixed)

        if not dedup.keep(text):
            continue

        # Saturating routing:
        # - If train full: everything goes to val buckets until each is full
        # - If a val bucket is full: its "val-assigned" items go to train
        if train_written >= args.target_train_tokens and not all_val_done():
            # force val fill
            if bucket in val_writers and val_written[bucket] < val_target:
                val_batch[bucket].append(text)
            else:
                # bucket doesn't have val writer or it's full; drop (do not keep reading forever)
                continue
        else:
            # normal hash split (but if val bucket full, route to train)
            split = assign_split(text, args.val_percent)
            if split == "val" and bucket in val_writers and val_written[bucket] < val_target:
                val_batch[bucket].append(text)
            else:
                train_batch.append(text)

        # Flush on batch size
        if len(train_batch) >= args.encode_batch_size:
            for ids in flush_text_batch(train_batch):
                if train_written < args.target_train_tokens:
                    push_ids_to_train(ids)
            train_batch = []

        for b in list(val_batch.keys()):
            if len(val_batch[b]) >= args.encode_batch_size:
                for ids in flush_text_batch(val_batch[b]):
                    if val_written[b] < val_target:
                        push_ids_to_val(b, ids)
                val_batch[b] = []

    # Final flush
    if train_batch and train_written < args.target_train_tokens:
        for ids in flush_text_batch(train_batch):
            if train_written < args.target_train_tokens:
                push_ids_to_train(ids)

    for b, batch in val_batch.items():
        if batch and val_written[b] < val_target:
            for ids in flush_text_batch(batch):
                if val_written[b] < val_target:
                    push_ids_to_val(b, ids)

    train_out.finalize()
    for w in val_writers.values():
        w.finalize()

    pbar_train.close()
    for p in pbar_val.values():
        p.close()

    print("✅ Phase2 v3 clean complete:")
    print(f"  train tokens: {train_written:,}")
    for k in sorted(val_written.keys()):
        print(f"  val {k}: {val_written[k]:,}")
    print(f"  out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()