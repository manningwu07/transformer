#!/usr/bin/env python3
import argparse
import hashlib
import json
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
    def __init__(self, name: str, it: Iterator[str], max_buffer: int = 64):
        self.name = name
        self._it = it
        self._q: queue.Queue[Optional[str]] = queue.Queue(maxsize=max_buffer)
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

    def try_get(self) -> Optional[str]:
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
) -> Iterator[str]:
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
        item = (item or "").strip()
        if item:
            yield item


# ----------------------------
# HF loader
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
    split: str,
    token: Optional[str],
    revision: Optional[str] = None,
):
    # IMPORTANT: do not map val->train here. Use hash split.
    kwargs = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision

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
# Split + formatting
# ----------------------------
def assign_split(text: str, val_percent: float) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, "little") % 10_000
    return "val" if x < int(val_percent * 100) else "train"


def format_chatml(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n"


def _safe_strip(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


# ----------------------------
# Synthetic hard-dependency episodes
# ----------------------------
def make_kv_retrieval_episode(
    tok: Tokenizer,
    rng: random.Random,
    min_tokens: int,
    max_tokens: int,
) -> str:
    header = (
        "You are given a long log of variable assignments.\n"
        "Rules:\n"
        "- Each line is VAR_xxxxxx = integer, or VAR_xxxxxx = VAR_yyyyyy + k.\n"
        "- Compute the requested VAR at the end.\n\n"
        "LOG BEGIN\n"
    )
    footer_tpl = "\nLOG END\n\nQuestion: What is {query_key}?\nAnswer with just the integer."

    target = rng.randint(min_tokens, max_tokens)
    n_vars = 0
    values: dict[str, int] = {}
    lines: list[str] = []

    for _ in range(64):
        k = f"VAR_{n_vars:06d}"
        v = rng.randint(-10_000_000, 10_000_000)
        values[k] = v
        lines.append(f"{k} = {v}")
        n_vars += 1

    query_key = f"VAR_{rng.randint(0, 32):06d}"

    while True:
        if rng.random() < 0.65:
            src = f"VAR_{rng.randint(0, n_vars - 1):06d}"
            dst = f"VAR_{n_vars:06d}"
            delta = rng.randint(-9999, 9999)
            values[dst] = values[src] + delta
            lines.append(f"{dst} = {src} + {delta}")
            n_vars += 1
        else:
            dst = f"VAR_{n_vars:06d}"
            v = rng.randint(-10_000_000, 10_000_000)
            values[dst] = v
            lines.append(f"{dst} = {v}")
            n_vars += 1

        if rng.random() < 0.35:
            src = f"VAR_{rng.randint(0, min(n_vars - 1, 128)):06d}"
            delta = rng.randint(-9999, 9999)
            values[query_key] = values[src] + delta
            lines.append(f"{query_key} = {src} + {delta}")

        if len(lines) % 128 == 0:
            body = "\n".join(lines)
            user = header + body + footer_tpl.format(query_key=query_key)
            if len(tok.encode(user).ids) >= target:
                return format_chatml(user, str(values[query_key]))


def harddeps_synth_iter(
    tok: Tokenizer,
    seed: int,
    min_tokens: int,
    max_tokens: int,
) -> Iterator[str]:
    rng = random.Random(seed)
    while True:
        yield make_kv_retrieval_episode(tok, rng, min_tokens, max_tokens)


# ----------------------------
# Real sources (streaming train only; split by hash)
# ----------------------------
def arxiv_iter(hf_token: Optional[str]) -> Iterator[str]:
    ds = load_dataset_streaming(
        "ccdv/arxiv-summarization",
        config=None,
        split="train",
        token=hf_token,
    )
    for ex in ds:
        article = _safe_strip(ex.get("article"))
        abstract = _safe_strip(ex.get("abstract"))
        if article and abstract:
            user = (
                "Summarize the following paper into a concise abstract.\n\n"
                f"<paper>\n{article}\n</paper>\n"
            )
            yield format_chatml(user, abstract)


def stack_edu_iter(hf_token: Optional[str], configs: List[str]) -> Iterator[str]:
    for cfg in configs:
        ds = load_dataset_streaming(
            "HuggingFaceTB/stack-edu",
            config=cfg,
            split="train",
            token=hf_token,
        )
        for ex in ds:
            t = ex.get("content") or ex.get("text") or ex.get("code") or ""
            t = _safe_strip(t)
            if t:
                yield t


def finemath_iter(hf_token: Optional[str], config: str) -> Iterator[str]:
    ds = load_dataset_streaming(
        "HuggingFaceTB/finemath",
        config=config,
        split="train",
        token=hf_token,
    )
    for ex in ds:
        t = _safe_strip(ex.get("text"))
        if t:
            yield t


# ----------------------------
# Encode to both splits (single pass + saturating targets)
# ----------------------------
def encode_stream_to_both_splits(
    text_iter: Iterator[str],
    tok: Tokenizer,
    eos_id: int,
    train_out: BinShardWriterFast,
    val_out: BinShardWriterFast,
    target_train_tokens: int,
    target_val_tokens: int,
    val_percent: float,
    encode_batch_size: int,
    min_item_tokens: int,
    max_item_tokens: int,
    dedup_capacity: int,
    desc: str,
) -> Tuple[int, int]:
    pbar_train = tqdm(total=target_train_tokens, unit="tok", desc=f"{desc} (train)")
    pbar_val = tqdm(total=target_val_tokens, unit="tok", desc=f"{desc} (val)")

    train_written = 0
    val_written = 0
    train_batch: List[str] = []
    val_batch: List[str] = []

    seen: set[str] = set()
    seen_added = 0

    def keep(text: str) -> bool:
        nonlocal seen, seen_added
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        if h in seen:
            return False
        seen.add(h)
        seen_added += 1
        if seen_added >= dedup_capacity:
            seen = set()
            seen_added = 0
        return True

    def process_ids(ids: List[int], is_val: bool) -> int:
        if not ids:
            return 0
        if ids[-1] != eos_id:
            ids.append(eos_id)
        if min_item_tokens > 0 and len(ids) < min_item_tokens:
            return 0
        if max_item_tokens > 0 and len(ids) > max_item_tokens:
            ids = ids[:max_item_tokens]
            ids[-1] = eos_id
        if is_val:
            val_out.push_ids(ids)
        else:
            train_out.push_ids(ids)
        return len(ids)

    def flush(batch: List[str], is_val: bool) -> int:
        if not batch:
            return 0
        encs = tok.encode_batch(batch)
        total = 0
        for enc in encs:
            total += process_ids(enc.ids, is_val)
        return total

    for t in text_iter:
        if train_written >= target_train_tokens and val_written >= target_val_tokens:
            break
        t = (t or "").strip()
        if not t or not keep(t):
            continue

        if train_written < target_train_tokens and val_written < target_val_tokens:
            split = assign_split(t, val_percent)
            (val_batch if split == "val" else train_batch).append(t)
        elif val_written < target_val_tokens:
            val_batch.append(t)
        elif train_written < target_train_tokens:
            train_batch.append(t)

        if len(train_batch) >= encode_batch_size:
            added = flush(train_batch, is_val=False)
            train_written += added
            pbar_train.update(added)
            train_batch = []

        if len(val_batch) >= encode_batch_size:
            added = flush(val_batch, is_val=True)
            val_written += added
            pbar_val.update(added)
            val_batch = []

    if train_batch and train_written < target_train_tokens:
        added = flush(train_batch, is_val=False)
        train_written += added
        pbar_train.update(added)

    if val_batch and val_written < target_val_tokens:
        added = flush(val_batch, is_val=True)
        val_written += added
        pbar_val.update(added)

    pbar_train.close()
    pbar_val.close()
    return train_written, val_written


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/longctx_hard_clean")
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_percent", type=float, default=2.0)

    ap.add_argument("--prefetch_buffer", type=int, default=64)
    ap.add_argument("--encode_batch_size", type=int, default=8)

    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)
    ap.add_argument("--target_train_tokens", type=int, default=350_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=25_000_000)

    ap.add_argument("--min_item_tokens", type=int, default=2048)
    ap.add_argument("--max_item_tokens", type=int, default=0, help="0 = no truncation")
    ap.add_argument("--dedup_capacity", type=int, default=300_000)

    ap.add_argument("--use_synth_harddeps", action="store_true")
    ap.add_argument("--use_arxiv", action="store_true")
    ap.add_argument("--use_stack_edu", action="store_true")
    ap.add_argument("--use_finemath", action="store_true")

    ap.add_argument("--finemath_config", type=str, default="finemath-3plus")
    ap.add_argument("--stack_edu_configs", type=str, default="Cpp,Go,JavaScript,Rust,TypeScript")

    ap.add_argument("--w_synth", type=float, default=0.40)
    ap.add_argument("--w_arxiv", type=float, default=0.30)
    ap.add_argument("--w_stack", type=float, default=0.20)
    ap.add_argument("--w_finemath", type=float, default=0.10)

    ap.add_argument("--synth_min_tokens", type=int, default=6500)
    ap.add_argument("--synth_max_tokens", type=int, default=9000)

    args = ap.parse_args()

    hf_token = _hf_token_from_env(args.hf_token)
    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")

    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32
    stack_cfgs = [c.strip() for c in args.stack_edu_configs.split(",") if c.strip()]

    # If user did not specify any --use_* flags, enable all by default.
    if not (args.use_synth_harddeps or args.use_arxiv or args.use_stack_edu or args.use_finemath):
        use_synth = use_arxiv = use_stack = use_finemath = True
    else:
        use_synth = args.use_synth_harddeps
        use_arxiv = args.use_arxiv
        use_stack = args.use_stack_edu
        use_finemath = args.use_finemath

    sources: List[Iterator[str]] = []
    probs: List[float] = []

    if use_synth:
        sources.append(
            harddeps_synth_iter(tok, seed=args.seed + 123, min_tokens=args.synth_min_tokens, max_tokens=args.synth_max_tokens)
        )
        probs.append(args.w_synth)
    if use_arxiv:
        sources.append(arxiv_iter(hf_token))
        probs.append(args.w_arxiv)
    if use_stack:
        sources.append(stack_edu_iter(hf_token, stack_cfgs))
        probs.append(args.w_stack)
    if use_finemath:
        sources.append(finemath_iter(hf_token, args.finemath_config))
        probs.append(args.w_finemath)

    if not sources:
        raise ValueError("No sources enabled.")

    total = sum(probs)
    probs = [p / max(1e-12, total) for p in probs]

    prefetch = [
        PrefetchIter(name=f"src_{i}", it=it, max_buffer=args.prefetch_buffer)
        for i, it in enumerate(sources)
    ]
    mixed = interleave_nonblocking(prefetch, probs=probs, seed=args.seed)

    train_out = BinShardWriterFast(
        out_dir=os.path.join(args.out_dir, "train"),
        prefix="longctx-hard-train",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )
    val_out = BinShardWriterFast(
        out_dir=os.path.join(args.out_dir, "val"),
        prefix="longctx-hard-val",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )

    wrote_train, wrote_val = encode_stream_to_both_splits(
        text_iter=mixed,
        tok=tok,
        eos_id=eos_id,
        train_out=train_out,
        val_out=val_out,
        target_train_tokens=args.target_train_tokens,
        target_val_tokens=args.target_val_tokens,
        val_percent=args.val_percent,
        encode_batch_size=args.encode_batch_size,
        min_item_tokens=args.min_item_tokens,
        max_item_tokens=args.max_item_tokens,
        dedup_capacity=args.dedup_capacity,
        desc="LongCtx Hard (clean)",
    )

    train_out.finalize()
    val_out.finalize()

    print("✅ longctx_hard_clean complete:")
    print(f"  train tokens: {wrote_train:,}")
    print(f"  val tokens:   {wrote_val:,}")
    print(f"  out_dir:      {args.out_dir}")


if __name__ == "__main__":
    main()