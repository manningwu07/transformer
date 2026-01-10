#!/usr/bin/env python3
import argparse
import os
import signal
import time
from dataclasses import dataclass
import traceback
from typing import Iterator, Optional, List

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )


def _timeout_handler(signum, frame):
    raise TimeoutError("HF next() timed out")


def next_with_timeout(it: Iterator[dict], timeout_sec: int) -> dict:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_sec))
    try:
        return next(it)
    finally:
        signal.alarm(0)


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
                self._buf.tofile(self._path())
                self.total_written += int(self._pos)
                self.shard_idx += 1
                self._pos = 0

    def finalize(self) -> None:
        if self._pos > 0:
            self._buf[: self._pos].tofile(self._path())
            self.total_written += int(self._pos)
            self.shard_idx += 1
            self._pos = 0


# Use the TechxGenus version which has 'text' field exposed properly
DATASET_NAME = "TechxGenus/stack-edu"

# Language filter values (lowercase to match dataset)
LANG_FILTER = {"cpp", "typescript", "go", "rust"}


def load_stackedu_iter(token: Optional[str]) -> Iterator[dict]:
    kwargs = {"split": "train", "streaming": True}
    if token:
        kwargs["token"] = token
    ds = load_dataset(DATASET_NAME, **kwargs)
    return iter(ds)


def extract_text(ex: dict) -> Optional[str]:
    # TechxGenus/stack-edu uses 'text' field
    v = ex.get("text", "")
    if isinstance(v, str) and v.strip():
        return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/shards/phase2_v2/val/stackedu",
    )
    ap.add_argument("--target_tokens", type=int, default=50_000_000)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument(
        "--languages",
        type=str,
        default="Cpp,TypeScript,Go,Rust",
        help="Comma-separated languages to filter (case-sensitive as in dataset)",
    )

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--timeout_sec", type=int, default=120)

    ap.add_argument(
        "--max_chars",
        type=int,
        default=200_000,
    )
    ap.add_argument("--encode_batch_size", type=int, default=8)
    ap.add_argument("--flush_interval_sec", type=float, default=2.0)

    ap.add_argument("--shard_size", type=int, default=50_000_000)
    args = ap.parse_args()

    hf_token = hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")

    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    # Parse target languages
    target_langs = {lang.strip() for lang in args.languages.split(",") if lang.strip()}
    print(f"Target languages: {target_langs}")

    print(f"→ Initializing {DATASET_NAME} (streaming)...", flush=True)
    it = load_stackedu_iter(hf_token)

    writer = ShardWriterFast(
        out_dir=args.out_dir,
        prefix="phase2-val-stackedu",
        shard_size=args.shard_size,
        dtype=dtype,
    )

    pbar = tqdm(total=args.target_tokens, unit="tok", desc="stackedu:val")
    written = 0
    batch: List[str] = []
    last_flush = time.time()
    skipped = 0
    processed = 0

    def flush_batch():
        nonlocal written, batch, last_flush
        if not batch:
            return
        encs = tok.encode_batch(batch)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)
            writer.push_ids(ids)
            written += len(ids)
            pbar.update(len(ids))
            if written >= args.target_tokens:
                break
        batch = []
        last_flush = time.time()

    while written < args.target_tokens:
        try:
            ex = next_with_timeout(it, args.timeout_sec)
        except StopIteration:
            print("\n⚠️ Dataset exhausted (StopIteration).", flush=True)
            break
        except TimeoutError:
            print("\n⚠️ Timeout. Reopening iterator...", flush=True)
            it = load_stackedu_iter(hf_token)
            continue
        except Exception as e:
            print(f"\n⚠️ Error: {type(e).__name__} repr={repr(e)}. Reopening...", flush=True)
            traceback.print_exc(limit=2)
            it = load_stackedu_iter(hf_token)
            continue

        processed += 1

        # Filter by language
        lang = ex.get("language", "")
        if lang not in target_langs:
            skipped += 1
            if processed % 100000 == 0:
                pbar.set_postfix({"skip": skipped, "proc": processed})
            continue

        t = extract_text(ex)
        if not t:
            continue
        t = t.strip()
        if args.max_chars > 0 and len(t) > args.max_chars:
            t = t[: args.max_chars]

        batch.append(t)
        now = time.time()
        if len(batch) >= args.encode_batch_size or (now - last_flush) >= args.flush_interval_sec:
            flush_batch()

    flush_batch()
    writer.finalize()
    pbar.close()
    print(f"✅ wrote {writer.total_written:,} tokens -> {args.out_dir}")
    print(f"   processed {processed:,} examples, skipped {skipped:,} (wrong language)")


if __name__ == "__main__":
    main()