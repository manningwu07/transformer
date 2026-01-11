#!/usr/bin/env python3
import argparse
import hashlib
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
    def __init__(
        self,
        name: str,
        it: Iterator[Tuple[str, str]],
        max_buffer: int = 64,
    ):
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
# HF loading + utils
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


def assign_split(text: str, val_percent: float) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, "little") % 10_000
    return "val" if x < int(val_percent * 100) else "train"


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


# ----------------------------
# Sources -> (bucket, text)
# ----------------------------
def finemath_texts(hf_token: Optional[str], config: str) -> Iterator[Tuple[str, str]]:
    ds = load_dataset_streaming("HuggingFaceTB/finemath", config=config, token=hf_token)
    for ex in ds:
        t = _safe_str(ex.get("text"))
        if t:
            yield "finemath", t


def stackedu_texts_techxgenus(
    hf_token: Optional[str],
    langs: set[str],
    max_chars: int,
) -> Iterator[Tuple[str, str]]:
    # This is the dataset you said worked reliably for you.
    ds = load_dataset_streaming("TechxGenus/stack-edu", config=None, token=hf_token)
    for ex in ds:
        lang = _safe_str(ex.get("language")).lower()
        if langs and lang not in langs:
            continue
        t = _safe_str(ex.get("text"))
        if not t:
            continue
        if max_chars > 0 and len(t) > max_chars:
            t = t[:max_chars]
        yield "stackedu", t


def anchor_texts(hf_token: Optional[str], name: str, config: str) -> Iterator[Tuple[str, str]]:
    ds = load_dataset_streaming(name, config=config, token=hf_token)
    for ex in ds:
        t = _safe_str(ex.get("text"))
        if t:
            yield "anchor", t


# ----------------------------
# Encoding helpers
# ----------------------------
def flush_text_batch(
    tok: Tokenizer,
    eos_id: int,
    texts: List[str],
) -> List[List[int]]:
    if not texts:
        return []
    encs = tok.encode_batch(texts)
    out: List[List[int]] = []
    for enc in encs:
        ids = enc.ids
        if not ids:
            continue
        if ids[-1] != eos_id:
            ids.append(eos_id)
        out.append(ids)
    return out


# ----------------------------
# Main: mixed train + per-bucket val, single run
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase2_v3_clean_techx")
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_percent", type=float, default=2.0)

    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)
    ap.add_argument("--target_train_tokens", type=int, default=4_500_000_000)
    ap.add_argument("--target_val_tokens_per_bucket", type=int, default=50_000_000)

    ap.add_argument("--prefetch_buffer", type=int, default=64)
    ap.add_argument("--encode_batch_size", type=int, default=512)

    # sources
    ap.add_argument("--finemath_config", type=str, default="finemath-4plus")

    ap.add_argument(
        "--stack_langs",
        type=str,
        default="cpp,typescript,go,rust",
        help="Comma-separated lowercase language names (TechxGenus uses language field).",
    )
    ap.add_argument("--stack_max_chars", type=int, default=200_000)

    ap.add_argument("--enable_anchor", action="store_true")
    ap.add_argument("--anchor_name", type=str, default="HuggingFaceTB/smollm-corpus")
    ap.add_argument("--anchor_config", type=str, default="fineweb-edu-dedup")

    ap.add_argument(
        "--train_probs",
        type=str,
        default="0.60,0.30,0.10",
        help="finemath,stackedu,anchor (if anchor disabled, still pass 2 values).",
    )

    args = ap.parse_args()
    hf_token = _hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")

    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    langs = {x.strip().lower() for x in args.stack_langs.split(",") if x.strip()}
    probs = [float(x) for x in args.train_probs.split(",") if x.strip()]

    if args.enable_anchor:
        if len(probs) != 3:
            raise ValueError("With --enable_anchor, --train_probs must have 3 values.")
    else:
        if len(probs) != 2:
            raise ValueError("Without anchor, --train_probs must have 2 values.")
        probs = [probs[0], probs[1], 0.0]

    s = sum(probs)
    probs = [p / max(1e-12, s) for p in probs]

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
    val_written = {k: 0 for k in val_writers.keys()}
    val_target = int(args.target_val_tokens_per_bucket)

    def all_val_done() -> bool:
        return all(val_written[k] >= val_target for k in val_writers.keys())

    pbar_train = tqdm(total=args.target_train_tokens, unit="tok", desc="phase2 (train)")
    pbar_val = {
        k: tqdm(total=val_target, unit="tok", desc=f"phase2 (val:{k})")
        for k in val_writers.keys()
    }

    # Sources
    sources: List[Iterator[Tuple[str, str]]] = []
    weights: List[float] = []

    sources.append(finemath_texts(hf_token, args.finemath_config))
    weights.append(probs[0])

    sources.append(stackedu_texts_techxgenus(hf_token, langs, args.stack_max_chars))
    weights.append(probs[1])

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

    train_batch: List[str] = []
    val_batch: Dict[str, List[str]] = {k: [] for k in val_writers.keys()}

    def push_train(ids: List[int]) -> None:
        nonlocal train_written
        train_out.push_ids(ids)
        train_written += len(ids)
        pbar_train.update(len(ids))

    def push_val(bucket: str, ids: List[int]) -> None:
        val_writers[bucket].push_ids(ids)
        val_written[bucket] += len(ids)
        pbar_val[bucket].update(len(ids))

    # Loop until both train + all val buckets are satisfied
    while train_written < args.target_train_tokens or not all_val_done():
        bucket, text = next(mixed)

        # decide split (hash) unless one side is saturated
        if train_written < args.target_train_tokens and not all_val_done():
            split = assign_split(text, args.val_percent)
            if split == "val" and bucket in val_writers and val_written[bucket] < val_target:
                val_batch[bucket].append(text)
            else:
                train_batch.append(text)
        elif train_written < args.target_train_tokens:
            # val done, force train
            train_batch.append(text)
        else:
            # train done, force val fill for the bucket if still needed; else drop
            if bucket in val_writers and val_written[bucket] < val_target:
                val_batch[bucket].append(text)
            else:
                continue

        if len(train_batch) >= args.encode_batch_size:
            for ids in flush_text_batch(tok, eos_id, train_batch):
                if train_written < args.target_train_tokens:
                    push_train(ids)
            train_batch = []

        for b in list(val_batch.keys()):
            if len(val_batch[b]) >= args.encode_batch_size:
                for ids in flush_text_batch(tok, eos_id, val_batch[b]):
                    if val_written[b] < val_target:
                        push_val(b, ids)
                val_batch[b] = []

    # Final flush
    if train_batch and train_written < args.target_train_tokens:
        for ids in flush_text_batch(tok, eos_id, train_batch):
            if train_written < args.target_train_tokens:
                push_train(ids)

    for b, batch in val_batch.items():
        if batch and val_written[b] < val_target:
            for ids in flush_text_batch(tok, eos_id, batch):
                if val_written[b] < val_target:
                    push_val(b, ids)

    train_out.finalize()
    for w in val_writers.values():
        w.finalize()

    pbar_train.close()
    for p in pbar_val.values():
        p.close()

    print("✅ Phase2 v3 clean (TechxGenus stack-edu) complete:")
    print(f"  train tokens: {train_written:,}")
    for k in sorted(val_written.keys()):
        print(f"  val {k}: {val_written[k]:,}")
    print(f"  out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()