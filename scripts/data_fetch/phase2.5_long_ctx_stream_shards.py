#!/usr/bin/env python3
import argparse
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


# ----------------------------
# Fast shard writer (no Python list)
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
        self.push_arr(arr)

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


# ----------------------------
# Nonblocking prefetch iterator (HF streaming can stall)
# ----------------------------
class PrefetchIter:
    def __init__(self, name: str, it: Iterator[str], max_buffer: int = 32):
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
    assert len(sources) == len(probs)
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
# HF loading (defensive across versions)
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
    kwargs = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision

    args = (name, config) if config is not None else (name,)

    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token

    # datasets has historically accepted token=... or use_auth_token=...
    if token:
        try:
            ds = load_dataset(*args, token=token, **kwargs)
        except TypeError:
            ds = load_dataset(*args, use_auth_token=token, **kwargs)
    else:
        ds = load_dataset(*args, **kwargs)

    return ds.with_format("python")


# ----------------------------
# Formatting + synthetic hard-dependency episodes
# ----------------------------
def format_chatml(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n"


def _safe_strip(s) -> str:
    return (s or "").strip() if isinstance(s, str) else ""


def make_kv_retrieval_episode(
    tok: Tokenizer,
    rng: random.Random,
    *,
    min_tokens: int,
    max_tokens: int,
) -> str:
    """
    Generates a long "needle / dependency" episode:
      - huge key/value log with chained references
      - question at the end depends on early keys
    This is deliberately position-sensitive training data for 8k+.
    """
    header = (
        "You are given a long log of variable assignments.\n"
        "Rules:\n"
        "- Each line is VAR_xxxxxx = integer, or VAR_xxxxxx = VAR_yyyyyy + k.\n"
        "- Compute the requested VAR at the end.\n\n"
        "LOG BEGIN\n"
    )
    footer_tpl = "\nLOG END\n\nQuestion: What is {query_key}?\nAnswer with just the integer."
    # Build progressively until token length hits target.
    target = rng.randint(min_tokens, max_tokens)
    n_vars = 0
    values: dict[str, int] = {}
    lines: list[str] = []

    # Seed a few base vars early (so query depends on early positions)
    for _ in range(64):
        k = f"VAR_{n_vars:06d}"
        v = rng.randint(-10_000_000, 10_000_000)
        values[k] = v
        lines.append(f"{k} = {v}")
        n_vars += 1

    # Choose an eventual query that depends on early vars (chain later)
    query_key = f"VAR_{rng.randint(0, 32):06d}"
    # Now extend log with long chains that reference earlier keys.
    while True:
        # Create a mix of direct and referenced assignments.
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

        # Make the query key drift later in the log via reassignment chains.
        if rng.random() < 0.35:
            # redefine query_key based on something old
            src = f"VAR_{rng.randint(0, min(n_vars - 1, 128)):06d}"
            delta = rng.randint(-9999, 9999)
            values[query_key] = values[src] + delta
            lines.append(f"{query_key} = {src} + {delta}")

        # Check token length occasionally.
        if len(lines) % 128 == 0:
            body = "\n".join(lines)
            text = header + body + footer_tpl.format(query_key=query_key)
            n_tok = len(tok.encode(text).ids)
            if n_tok >= target:
                answer = str(values[query_key])
                return format_chatml(text, answer)


def harddeps_synth_iter(
    tok: Tokenizer,
    seed: int,
    min_tokens: int,
    max_tokens: int,
) -> Iterator[str]:
    rng = random.Random(seed)
    while True:
        yield make_kv_retrieval_episode(
            tok,
            rng,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )


# ----------------------------
# Real sources
# ----------------------------
def arxiv_summarization_iter(hf_token: Optional[str], split: str) -> Iterator[str]:
    ds = load_dataset_streaming(
        "ccdv/arxiv-summarization",
        config=None,
        split=split,
        token=hf_token,
    )
    for ex in ds:
        article = _safe_strip(ex.get("article", ""))
        abstract = _safe_strip(ex.get("abstract", ""))
        if not article or not abstract:
            continue
        user = (
            "Summarize the following paper into a concise abstract.\n\n"
            f"<paper>\n{article}\n</paper>\n"
        )
        yield format_chatml(user, abstract)


def stack_edu_iter(hf_token: Optional[str], split: str, configs: List[str]) -> Iterator[str]:
    for cfg in configs:
        ds = load_dataset_streaming(
            "HuggingFaceTB/stack-edu",
            config=cfg,
            split=split,
            token=hf_token,
        )
        for ex in ds:
            t = ex.get("content") or ex.get("text") or ex.get("code") or ""
            t = _safe_strip(t)
            if t:
                yield t


def finemath_iter(hf_token: Optional[str], split: str) -> Iterator[str]:
    ds = load_dataset_streaming(
        "HuggingFaceTB/finemath",
        config="finemath-3plus",
        split=split,
        token=hf_token,
    )
    for ex in ds:
        t = _safe_strip(ex.get("text", ""))
        if t:
            yield t


# ----------------------------
# Encoding loop
# ----------------------------
def encode_text_stream_to_bin(
    text_iter: Iterator[str],
    tok: Tokenizer,
    eos_id: int,
    out: BinShardWriterFast,
    target_tokens: int,
    encode_batch_size: int,
    flush_interval_sec: float,
    min_item_tokens: int,
    desc: str,
) -> int:
    pbar = tqdm(total=target_tokens, unit="tok", desc=desc)
    written = 0
    batch: List[str] = []
    last_flush_t = time.time()

    def flush(batch_texts: List[str]) -> None:
        nonlocal written
        encs = tok.encode_batch(batch_texts)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)
            if min_item_tokens > 0 and len(ids) < min_item_tokens:
                continue
            out.push_ids(ids)
            written += len(ids)
            pbar.update(len(ids))
            if written >= target_tokens:
                return

    for t in text_iter:
        t = (t or "").strip()
        if not t:
            continue
        batch.append(t)
        now = time.time()
        if len(batch) >= encode_batch_size or (now - last_flush_t) >= flush_interval_sec:
            flush(batch)
            batch = []
            last_flush_t = now
            if written >= target_tokens:
                break

    if batch and written < target_tokens:
        flush(batch)

    pbar.close()
    return written


def build_split(
    *,
    split: str,
    tok: Tokenizer,
    eos_id: int,
    out_dir: str,
    shard_size: int,
    dtype: np.dtype,
    target_tokens: int,
    seed: int,
    hf_token: Optional[str],
    prefetch_buffer: int,
    encode_batch_size: int,
    flush_interval_sec: float,
    min_item_tokens: int,
    # source toggles
    use_synth_harddeps: bool,
    use_arxiv: bool,
    use_stack_edu: bool,
    use_finemath: bool,
    stack_edu_configs: List[str],
    # weights
    w_synth: float,
    w_arxiv: float,
    w_stack: float,
    w_finemath: float,
    synth_min_tokens: int,
    synth_max_tokens: int,
) -> int:
    out = BinShardWriterFast(
        out_dir=os.path.join(out_dir, split),
        prefix=f"phase3-longctx-hard-{split}",
        shard_size=shard_size,
        dtype=dtype,
    )

    text_sources: list[Iterator[str]] = []
    probs: list[float] = []

    if use_synth_harddeps:
        text_sources.append(
            harddeps_synth_iter(
                tok,
                seed=seed + 123,
                min_tokens=synth_min_tokens,
                max_tokens=synth_max_tokens,
            )
        )
        probs.append(w_synth)

    if use_arxiv:
        text_sources.append(arxiv_summarization_iter(hf_token, split=split))
        probs.append(w_arxiv)

    if use_stack_edu:
        text_sources.append(stack_edu_iter(hf_token, split=split, configs=stack_edu_configs))
        probs.append(w_stack)

    if use_finemath:
        text_sources.append(finemath_iter(hf_token, split=split))
        probs.append(w_finemath)

    if not text_sources:
        raise ValueError("No sources enabled.")

    total = sum(probs)
    probs = [p / max(1e-12, total) for p in probs]

    prefetch = [
        PrefetchIter(name=f"{split}_{i}", it=it, max_buffer=prefetch_buffer)
        for i, it in enumerate(text_sources)
    ]
    mixed = interleave_nonblocking(prefetch, probs=probs, seed=seed)

    wrote = encode_text_stream_to_bin(
        text_iter=mixed,
        tok=tok,
        eos_id=eos_id,
        out=out,
        target_tokens=target_tokens,
        encode_batch_size=encode_batch_size,
        flush_interval_sec=flush_interval_sec,
        min_item_tokens=min_item_tokens,
        desc=f"Phase3 LongCtx HardDeps ({split})",
    )
    out.finalize()
    return wrote


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase3_longctx_hard")
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prefetch_buffer", type=int, default=32)
    ap.add_argument("--encode_batch_size", type=int, default=6)
    ap.add_argument("--flush_interval_sec", type=float, default=2.0)

    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)
    ap.add_argument("--target_train_tokens", type=int, default=350_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=25_000_000)

    ap.add_argument(
        "--min_item_tokens",
        type=int,
        default=2048,
        help="Drop items shorter than this (forces long-context exposure).",
    )

    # sources
    ap.add_argument("--use_synth_harddeps", action="store_true")
    ap.add_argument("--use_arxiv", action="store_true")
    ap.add_argument("--use_stack_edu", action="store_true")
    ap.add_argument("--use_finemath", action="store_true")

    ap.add_argument(
        "--stack_edu_configs",
        type=str,
        default="Python,Cpp,Go,JavaScript,Rust",
    )

    # weights
    ap.add_argument("--w_synth", type=float, default=0.40)
    ap.add_argument("--w_arxiv", type=float, default=0.30)
    ap.add_argument("--w_stack", type=float, default=0.20)
    ap.add_argument("--w_finemath", type=float, default=0.10)

    # synth episode sizing (token-count in a single sample)
    ap.add_argument("--synth_min_tokens", type=int, default=6500)
    ap.add_argument("--synth_max_tokens", type=int, default=9000)

    args = ap.parse_args()

    hf_token = _hf_token_from_env(args.hf_token)
    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <|endoftext|> token.")
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    stack_cfgs = [c.strip() for c in args.stack_edu_configs.split(",") if c.strip()]

    # Sensible defaults if you forget flags: enable the “hard” sources
    use_synth = bool(args.use_synth_harddeps or True)
    use_arxiv = bool(args.use_arxiv or True)
    use_stack = bool(args.use_stack_edu or True)
    use_finemath = bool(args.use_finemath or True)

    print("Phase3 LongCtx HardDeps:")
    print(f"  out_dir: {args.out_dir}")
    print(f"  dtype:   {dtype}")
    print(
        "  sources:",
        f"synth={int(use_synth)} arxiv={int(use_arxiv)} "
        f"stack={int(use_stack)} finemath={int(use_finemath)}",
    )
    print(
        "  weights:",
        f"synth={args.w_synth} arxiv={args.w_arxiv} "
        f"stack={args.w_stack} finemath={args.w_finemath}",
    )

    wrote_train = build_split(
        split="train",
        tok=tok,
        eos_id=eos_id,
        out_dir=args.out_dir,
        shard_size=args.bin_shard_size,
        dtype=dtype,
        target_tokens=args.target_train_tokens,
        seed=args.seed,
        hf_token=hf_token,
        prefetch_buffer=args.prefetch_buffer,
        encode_batch_size=args.encode_batch_size,
        flush_interval_sec=args.flush_interval_sec,
        min_item_tokens=args.min_item_tokens,
        use_synth_harddeps=use_synth,
        use_arxiv=use_arxiv,
        use_stack_edu=use_stack,
        use_finemath=use_finemath,
        stack_edu_configs=stack_cfgs,
        w_synth=args.w_synth,
        w_arxiv=args.w_arxiv,
        w_stack=args.w_stack,
        w_finemath=args.w_finemath,
        synth_min_tokens=args.synth_min_tokens,
        synth_max_tokens=args.synth_max_tokens,
    )
    wrote_val = build_split(
        split="val",
        tok=tok,
        eos_id=eos_id,
        out_dir=args.out_dir,
        shard_size=args.bin_shard_size,
        dtype=dtype,
        target_tokens=args.target_val_tokens,
        seed=args.seed + 999,
        hf_token=hf_token,
        prefetch_buffer=args.prefetch_buffer,
        encode_batch_size=max(1, args.encode_batch_size // 2),
        flush_interval_sec=args.flush_interval_sec,
        min_item_tokens=args.min_item_tokens,
        use_synth_harddeps=use_synth,
        use_arxiv=use_arxiv,
        use_stack_edu=use_stack,
        use_finemath=use_finemath,
        stack_edu_configs=stack_cfgs,
        w_synth=args.w_synth,
        w_arxiv=args.w_arxiv,
        w_stack=args.w_stack,
        w_finemath=args.w_finemath,
        synth_min_tokens=args.synth_min_tokens,
        synth_max_tokens=args.synth_max_tokens,
    )

    print("✅ Phase3 LongCtx HardDeps complete:")
    print(f"  train tokens: {wrote_train:,}")
    print(f"  val tokens:   {wrote_val:,}")
    print(f"  out_dir:      {args.out_dir}")


if __name__ == "__main__":
    main()