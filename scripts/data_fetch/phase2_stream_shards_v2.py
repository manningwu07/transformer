#!/usr/bin/env python3
"""
Phase 2 Sharder (v2 fixed)

Train:
  - Mixed stream of: FineMath + Stack-Edu (no Python)
  - Optional: generalist anchor (e.g., SmolLM FineWeb-Edu-dedup)

Val:
  - Per-source val shards:
      out_dir/val/finemath/*.bin
      out_dir/val/stackedu/*.bin
      out_dir/val/anchor/*.bin   (if --enable_anchor)

Key fix vs prior v2:
  - All signal-based timeout priming happens in MAIN THREAD only.
  - Prefetch threads never call signal.signal / signal.alarm.
"""

import argparse
import os
import random
import signal
import threading
import time
import queue
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


# -------------------- HF token --------------------
def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


# -------------------- Timeout priming (MAIN THREAD ONLY) --------------------
def _timeout_handler(signum, frame):
    raise TimeoutError("HF streaming prime timed out")


def prime_streaming_iterator(
    name: str,
    config: Optional[str],
    split: str,
    token: Optional[str],
    timeout_sec: int,
) -> Iterator[dict]:
    """
    Load a streaming dataset and "prime" it by pulling the first item with a
    SIGALRM timeout. MUST be called in the main thread.

    Returns an iterator that yields the first item, then the rest.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "prime_streaming_iterator must be called from the main thread."
        )

    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token

    args = (name, config) if config else (name,)
    print(f"  → Loading {name}" + (f" [{config}]" if config else "") + "...", flush=True)
    ds = load_dataset(*args, **kwargs).with_format("python")
    it = iter(ds)

    # SIGALRM exists on macOS/Linux; if it doesn't, this will throw.
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_sec))
    try:
        first = next(it)
        print(
            f"    ✓ Primed {name}" + (f" [{config}]" if config else ""),
            flush=True,
        )

        def _gen():
            yield first
            yield from it

        return _gen()
    except Exception as e:
        print(
            f"    ✗ Failed prime {name}"
            + (f" [{config}]" if config else "")
            + f": {e}",
            flush=True,
        )
        return iter([])
    finally:
        signal.alarm(0)


# -------------------- Prefetch (prevents stalls) --------------------
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
        if not self._done.is_set():
            return True
        return not self._q.empty()


def interleave_nonblocking(
    sources: List[PrefetchIter],
    probs: List[float],
    seed: int,
    idle_sleep_sec: float = 0.01,
) -> Iterator[str]:
    assert len(sources) == len(probs)
    rng = random.Random(seed)

    while any(s.alive() for s in sources):
        available = []
        weights = []
        for i, s in enumerate(sources):
            if probs[i] <= 0:
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
        yield item
        
def prefetch_single(it: Iterator[str], seed: int) -> Iterator[str]:
    """
    Wrap a text iterator in PrefetchIter and yield from it using the same
    nonblocking pattern (prevents main-thread HF stalls).
    """
    src = PrefetchIter("single", it, max_buffer=64)
    for t in interleave_nonblocking([src], probs=[1.0], seed=seed):
        t = (t or "").strip()
        if t:
            yield t


# -------------------- Sharding --------------------
@dataclass
class ShardState:
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

    def push_ids(self, ids: List[int]) -> None:
        if not ids:
            return
        self.buf.extend(ids)
        self._flush_full()

    def _flush_full(self) -> None:
        while len(self.buf) >= self.shard_size:
            chunk = self.buf[: self.shard_size]
            arr = np.asarray(chunk, dtype=self.dtype)
            arr.tofile(self._path())
            self.total_written += int(arr.size)
            self.shard_idx += 1
            self.buf = self.buf[self.shard_size :]

    def finalize(self) -> None:
        if self.buf:
            arr = np.asarray(self.buf, dtype=self.dtype)
            arr.tofile(self._path())
            self.total_written += int(arr.size)
            self.shard_idx += 1
            self.buf = []


def encode_text_stream_to_shards(
    text_iter: Iterator[str],
    tokenizer: Tokenizer,
    eos_id: int,
    state: ShardState,
    target_tokens: int,
    seed: int,
    encode_batch_size: int,
    flush_interval_sec: float,
    desc: str,
) -> int:
    rng = random.Random(seed)
    pbar = tqdm(total=int(target_tokens), unit="tok", desc=desc)
    written = 0
    batch: List[str] = []
    last_flush = time.time()

    def flush_batch(texts: List[str]) -> None:
        nonlocal written, last_flush
        encs = tokenizer.encode_batch(texts)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)
            state.push_ids(ids)
            written += len(ids)
            pbar.update(len(ids))
            if written >= target_tokens:
                return
        last_flush = time.time()

    empty_streak = 0
    for t in text_iter:
        t = (t or "").strip()
        if not t:
            empty_streak += 1
            if empty_streak >= 10000:
                print(f"⚠️ {desc}: 10k empty docs streak", flush=True)
                empty_streak = 0
            continue
        empty_streak = 0

        batch.append(t)
        now = time.time()
        if len(batch) >= encode_batch_size or (now - last_flush) >= flush_interval_sec:
            flush_batch(batch)
            batch = []
            if written >= target_tokens:
                break

    if batch and written < target_tokens:
        flush_batch(batch)

    pbar.close()
    return int(written)


# -------------------- Source generators (consume primed iterators) --------------------
def finemath_text_from_iter(it: Iterator[dict]) -> Iterator[str]:
    for ex in it:
        t = ex.get("text", "")
        if isinstance(t, str) and t.strip():
            yield t


def smollm_text_from_iter(it: Iterator[dict]) -> Iterator[str]:
    for ex in it:
        t = ex.get("text", "")
        if isinstance(t, str) and t.strip():
            yield t


def stackedu_text_from_iters(cfg_iters: List[Tuple[str, Iterator[dict]]]) -> Iterator[str]:
    """
    Uniformly mix multiple Stack-Edu config iterators.
    Assumes each iterator is already primed in main thread.
    """
    rng = random.Random(1337)
    active = [True] * len(cfg_iters)
    buffers: List[Optional[dict]] = [None] * len(cfg_iters)

    for i, (_, it) in enumerate(cfg_iters):
        try:
            buffers[i] = next(it)
        except StopIteration:
            active[i] = False
            buffers[i] = None

    keys = ["content", "text", "code"]
    while any(active):
        avail = [i for i in range(len(cfg_iters)) if active[i]]
        if not avail:
            break
        i = rng.choice(avail)

        ex = buffers[i]
        if ex is not None:
            t = ""
            for k in keys:
                v = ex.get(k, "")
                if isinstance(v, str) and v.strip():
                    t = v
                    break
            if t:
                yield t

        try:
            buffers[i] = next(cfg_iters[i][1])
        except StopIteration:
            active[i] = False
            buffers[i] = None


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase2_v2")

    ap.add_argument("--target_train_tokens", type=int, default=4_500_000_000)
    ap.add_argument(
        "--target_val_tokens_per_source",
        type=int,
        default=50_000_000,
        help="Each of finemath/stackedu (and anchor if enabled) gets this many val tokens",
    )

    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--encode_batch_size", type=int, default=512)
    ap.add_argument("--flush_interval_sec", type=float, default=2.0)

    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--timeout_sec", type=int, default=120)

    # Phase2 sources
    ap.add_argument(
        "--finemath_config",
        type=str,
        default="finemath-4plus",
    )
    ap.add_argument(
        "--stack_edu_configs",
        type=str,
        default="TypeScript,Go,Cpp,Rust",
    )

    # Optional generalist anchor
    ap.add_argument("--enable_anchor", action="store_true")
    ap.add_argument(
        "--anchor_name",
        type=str,
        default="HuggingFaceTB/smollm-corpus",
    )
    ap.add_argument(
        "--anchor_config",
        type=str,
        default="fineweb-edu-dedup",
        help="For smollm-corpus, a good generalist anchor is fineweb-edu-dedup.",
    )

    # Train mixture weights
    ap.add_argument(
        "--train_probs",
        type=str,
        default="0.60,0.40",
        help=(
            "If no anchor: finemath,stackedu. "
            "If --enable_anchor: finemath,stackedu,anchor."
        ),
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
        raise ValueError("Do not include Python in --stack_edu_configs.")

    probs = [float(x) for x in args.train_probs.split(",") if x.strip()]
    want = 3 if args.enable_anchor else 2
    if len(probs) != want:
        raise ValueError(f"--train_probs must have {want} values.")
    s = sum(probs)
    if s <= 0:
        raise ValueError("train_probs sum must be > 0")
    probs = [p / s for p in probs]

    print("=" * 60)
    print("Phase 2 v2 (fixed)")
    print(f"  FineMath config: {args.finemath_config}")
    print(f"  Stack-Edu configs: {stack_cfgs}")
    if args.enable_anchor:
        print(f"  Anchor: {args.anchor_name} [{args.anchor_config}]")
    print(f"  Train probs: {probs}")
    print(f"  Val tokens / source: {args.target_val_tokens_per_source:,}")
    print(f"  Out: {args.out_dir}")
    print("=" * 60, flush=True)

    # ---- PRIME IN MAIN THREAD (fixes your crash) ----
    fm_it = prime_streaming_iterator(
        "HuggingFaceTB/finemath",
        args.finemath_config,
        split="train",
        token=hf_token,
        timeout_sec=args.timeout_sec,
    )

    se_cfg_iters: List[Tuple[str, Iterator[dict]]] = []
    for cfg in stack_cfgs:
        it = prime_streaming_iterator(
            "HuggingFaceTB/stack-edu",
            cfg,
            split="train",
            token=hf_token,
            timeout_sec=args.timeout_sec,
        )
        se_cfg_iters.append((cfg, it))

    anchor_it: Optional[Iterator[dict]] = None
    if args.enable_anchor:
        anchor_it = prime_streaming_iterator(
            args.anchor_name,
            args.anchor_config,
            split="train",
            token=hf_token,
            timeout_sec=args.timeout_sec,
        )

    # ---- TRAIN (mixed) ----
    fm_train = PrefetchIter("finemath", finemath_text_from_iter(fm_it), max_buffer=64)
    se_train = PrefetchIter(
        "stackedu",
        stackedu_text_from_iters(se_cfg_iters),
        max_buffer=64,
    )

    sources = [fm_train, se_train]
    if args.enable_anchor:
        assert anchor_it is not None
        sources.append(PrefetchIter("anchor", smollm_text_from_iter(anchor_it), 64))

    train_text = interleave_nonblocking(sources=sources, probs=probs, seed=args.seed)

    train_state = ShardState(
        out_dir=os.path.join(args.out_dir, "train"),
        prefix="phase2-train-mixed",
        shard_size=args.shard_size,
        dtype=dtype,
    )
    train_written = encode_text_stream_to_shards(
        text_iter=train_text,
        tokenizer=tok,
        eos_id=eos_id,
        state=train_state,
        target_tokens=args.target_train_tokens,
        seed=args.seed,
        encode_batch_size=args.encode_batch_size,
        flush_interval_sec=args.flush_interval_sec,
        desc="phase2:train(mixed)",
    )
    train_state.finalize()
    print(f"Train: {train_written:,} tokens, shards={train_state.shard_idx}", flush=True)

    # ---- VAL (per-source, clean signal) ----
    # FineMath val
    fm_val_state = ShardState(
        out_dir=os.path.join(args.out_dir, "val", "finemath"),
        prefix="phase2-val-finemath",
        shard_size=args.shard_size,
        dtype=dtype,
    )
    fm_val_written = encode_text_stream_to_shards(
        text_iter=prefetch_single(
            finemath_text_from_iter(
                prime_streaming_iterator(
                 "HuggingFaceTB/finemath",
                 args.finemath_config,
                 split="train",
                 token=hf_token,
                 timeout_sec=args.timeout_sec,
                )
            ),
            seed=args.seed + 101,
        ),
         tokenizer=tok,
         eos_id=eos_id,
         state=fm_val_state,
         target_tokens=args.target_val_tokens_per_source,
         seed=args.seed + 101,
        encode_batch_size=args.encode_batch_size,
        flush_interval_sec=args.flush_interval_sec,
        desc="phase2:val(finemath)",
    )
    fm_val_state.finalize()

    # Stack-Edu val
    se_val_state = ShardState(
        out_dir=os.path.join(args.out_dir, "val", "stackedu"),
        prefix="phase2-val-stackedu",
        shard_size=args.shard_size,
        dtype=dtype,
    )
    def stackedu_text_single(it: Iterator[dict]) -> Iterator[str]:
        keys = ["content", "text", "code"]
        for ex in it:
            for k in keys:
                v = ex.get(k, "")
                if isinstance(v, str) and v.strip():
                    yield v
                    break

    se_val_sources: List[PrefetchIter] = []
    for cfg in stack_cfgs:
        it = prime_streaming_iterator(
            "HuggingFaceTB/stack-edu",
            cfg,
            split="train",
            token=hf_token,
            timeout_sec=args.timeout_sec,
        )
        se_val_sources.append(
            PrefetchIter(f"stackedu_{cfg}", stackedu_text_single(it), max_buffer=64)
        )

    se_val_text = interleave_nonblocking(
        sources=se_val_sources,
        probs=[1.0 / len(se_val_sources)] * len(se_val_sources),
        seed=args.seed + 202,
    )

    se_val_written = encode_text_stream_to_shards(
        text_iter=se_val_text,
        tokenizer=tok,
        eos_id=eos_id,
        state=se_val_state,
        target_tokens=args.target_val_tokens_per_source,
        seed=args.seed + 202,
        encode_batch_size=args.encode_batch_size,
        flush_interval_sec=args.flush_interval_sec,
        desc="phase2:val(stackedu)",
    )
    se_val_state.finalize()

    anchor_val_written = 0
    if args.enable_anchor:
        anchor_val_state = ShardState(
            out_dir=os.path.join(args.out_dir, "val", "anchor"),
            prefix="phase2-val-anchor",
            shard_size=args.shard_size,
            dtype=dtype,
        )
        anchor_val_written = encode_text_stream_to_shards(
            text_iter=prefetch_single(
                smollm_text_from_iter(
                    prime_streaming_iterator(
                        args.anchor_name,
                        args.anchor_config,
                        split="train",
                        token=hf_token,
                        timeout_sec=args.timeout_sec,
                    )
                ),
                seed=args.seed + 303,
            ),
            tokenizer=tok,
            eos_id=eos_id,
            state=anchor_val_state,
            target_tokens=args.target_val_tokens_per_source,
            seed=args.seed + 303,
            encode_batch_size=args.encode_batch_size,
            flush_interval_sec=args.flush_interval_sec,
            desc="phase2:val(anchor)",
        )
        anchor_val_state.finalize()

    print("=" * 60)
    print("Done.")
    print(f"  Train mixed: {train_written:,}")
    print(f"  Val finemath: {fm_val_written:,}")
    print(f"  Val stacked u: {se_val_written:,}")
    if args.enable_anchor:
        print(f"  Val anchor: {anchor_val_written:,}")
    print(f"  Output: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()