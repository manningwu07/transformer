#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import queue
import random
import re
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional, List, Dict, Any, Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="datasets")


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
    kwargs["trust_remote_code"] = True

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


def format_chatml_turn(role: str, content: str) -> str:
    role = role.strip().lower()
    if role not in {"system", "user", "assistant"}:
        role = "user"
    content = (content or "").strip()
    return f"<|{role}|>\n{content}\n"


def format_chatml_dialog(turns: List[Tuple[str, str]]) -> Optional[str]:
    out = []
    for role, content in turns:
        content = (content or "").strip()
        if not content:
            continue
        out.append(format_chatml_turn(role, content))
    if not out:
        return None
    return "".join(out)


def normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in {"human", "user"}:
        return "user"
    if r in {"assistant", "gpt", "model"}:
        return "assistant"
    if r in {"system"}:
        return "system"
    return "user"


def example_to_chatml(ex: Dict[str, Any]) -> Optional[str]:
    if isinstance(ex.get("conversations"), list):
        turns = []
        for t in ex["conversations"]:
            if not isinstance(t, dict):
                continue
            role = normalize_role(t.get("from", "user"))
            val = t.get("value", "")
            if isinstance(val, str) and val.strip():
                turns.append((role, val))
        return format_chatml_dialog(turns)

    if isinstance(ex.get("messages"), list):
        turns = []
        for t in ex["messages"]:
            if not isinstance(t, dict):
                continue
            role = normalize_role(t.get("role", "user"))
            val = t.get("content", "")
            if isinstance(val, str) and val.strip():
                turns.append((role, val))
        return format_chatml_dialog(turns)

    inst = ex.get("instruction")
    out = ex.get("output")
    if isinstance(inst, str) and isinstance(out, str) and inst.strip() and out.strip():
        inp = ex.get("input")
        user = inst.strip()
        if isinstance(inp, str) and inp.strip():
            user = f"{user}\n\nInput:\n{inp.strip()}"
        return format_chatml_dialog([("user", user), ("assistant", out)])

    prompt = ex.get("prompt")
    resp = ex.get("response")
    if isinstance(prompt, str) and isinstance(resp, str) and prompt.strip() and resp.strip():
        return format_chatml_dialog([("user", prompt), ("assistant", resp)])

    return None


def format_glaive_v2(ex: dict) -> str:
    text = (ex.get("chat", "") or "").strip()
    if not text:
        return ""

    text = text.replace("SYSTEM: ", "<|system|>\n")
    text = text.replace("USER: ", "\n<|user|>\n")
    text = text.replace("ASSISTANT: ", "\n<|assistant|>\n")
    text = text.replace("<|endoftext|>", "")

    text = re.sub(
        r"FUNCTION RESPONSE:\s*(.*?)(?=\n<\|user\|>|\n<\|assistant\|>|\n<\|system\|>|\Z)",
        r"<response>\1</response>",
        text,
        flags=re.DOTALL,
    )

    def json_call_replacer(m):
        raw = m.group(0)
        try:
            obj = json.loads(raw)
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            tag = "call:search" if "search" in str(name).lower() else "call:tool"
            payload = {"name": name, "args": args}
            return f"<{tag}>{json.dumps(payload)}</call>"
        except Exception:
            return raw

    text = re.sub(
        r'\{"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*.*?\}',
        json_call_replacer,
        text,
        flags=re.DOTALL,
    )
    return text.strip() + "\n"


def assign_split(text: str, val_percent: float) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, "little") % 10_000
    return "val" if x < int(val_percent * 100) else "train"


def iter_generic_chat(
    dataset: str,
    config: Optional[str],
    hf_token: Optional[str],
) -> Iterator[str]:
    ds = load_dataset_streaming(dataset, config=config, split="train", token=hf_token)
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        s = example_to_chatml(ex)
        if s:
            yield s


def iter_glaive(hf_token: Optional[str]) -> Iterator[str]:
    ds = load_dataset_streaming(
        "glaiveai/glaive-function-calling-v2",
        config=None,
        split="train",
        token=hf_token,
    )
    for ex in ds:
        s = format_glaive_v2(ex)
        if s:
            yield s


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
) -> Tuple[int, int]:
    pbar_train = tqdm(total=target_train_tokens, unit="tok", desc="Phase3 SFT (train)")
    pbar_val = tqdm(total=target_val_tokens, unit="tok", desc="Phase3 SFT (val)")

    train_written = 0
    val_written = 0

    train_batch: List[str] = []
    val_batch: List[str] = []

    seen: set[str] = set()
    seen_added = 0

    def keep(text: str) -> bool:
        nonlocal seen_added, seen
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
            if ids[-1] != eos_id:
                ids[-1] = eos_id

        if is_val:
            val_out.push_ids(ids)
        else:
            train_out.push_ids(ids)
        return len(ids)

    def flush_batch(batch: List[str], is_val: bool) -> int:
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

        # KEY FIX: Saturate on overflow - once one bucket is full, send everything to the other
        if train_written < target_train_tokens and val_written < target_val_tokens:
            split = assign_split(t, val_percent)
            if split == "val":
                val_batch.append(t)
            else:
                train_batch.append(t)
        elif val_written < target_val_tokens:
            val_batch.append(t)
        elif train_written < target_train_tokens:
            train_batch.append(t)

        if len(train_batch) >= encode_batch_size:
            added = flush_batch(train_batch, is_val=False)
            train_written += added
            pbar_train.update(added)
            train_batch = []

        if len(val_batch) >= encode_batch_size:
            added = flush_batch(val_batch, is_val=True)
            val_written += added
            pbar_val.update(added)
            val_batch = []

    if train_batch and train_written < target_train_tokens:
        added = flush_batch(train_batch, is_val=False)
        train_written += added
        pbar_train.update(added)

    if val_batch and val_written < target_val_tokens:
        added = flush_batch(val_batch, is_val=True)
        val_written += added
        pbar_val.update(added)

    pbar_train.close()
    pbar_val.close()

    return train_written, val_written


def build_both_splits(
    *,
    tok: Tokenizer,
    eos_id: int,
    out_dir: str,
    shard_size: int,
    dtype: np.dtype,
    target_train_tokens: int,
    target_val_tokens: int,
    seed: int,
    hf_token: Optional[str],
    prefetch_buffer: int,
    encode_batch_size: int,
    min_item_tokens: int,
    max_item_tokens: int,
    dedup_capacity: int,
    generic_dataset: str,
    generic_config: Optional[str],
    val_percent: float,
    use_glaive: bool,
    w_generic: float,
    w_glaive: float,
) -> Tuple[int, int]:
    train_out = BinShardWriterFast(
        out_dir=os.path.join(out_dir, "train"),
        prefix="phase3-sft-generic-train",
        shard_size=shard_size,
        dtype=dtype,
    )
    val_out = BinShardWriterFast(
        out_dir=os.path.join(out_dir, "val"),
        prefix="phase3-sft-generic-val",
        shard_size=shard_size,
        dtype=dtype,
    )

    sources: list[Iterator[str]] = []
    probs: list[float] = []

    sources.append(iter_generic_chat(generic_dataset, generic_config, hf_token))
    probs.append(w_generic)

    if use_glaive:
        sources.append(iter_glaive(hf_token))
        probs.append(w_glaive)

    total = sum(probs)
    probs = [p / max(1e-12, total) for p in probs]

    prefetch = [
        PrefetchIter(name=f"src_{i}", it=it, max_buffer=prefetch_buffer)
        for i, it in enumerate(sources)
    ]
    mixed = interleave_nonblocking(prefetch, probs=probs, seed=seed)

    train_wrote, val_wrote = encode_stream_to_both_splits(
        text_iter=mixed,
        tok=tok,
        eos_id=eos_id,
        train_out=train_out,
        val_out=val_out,
        target_train_tokens=target_train_tokens,
        target_val_tokens=target_val_tokens,
        val_percent=val_percent,
        encode_batch_size=encode_batch_size,
        min_item_tokens=min_item_tokens,
        max_item_tokens=max_item_tokens,
        dedup_capacity=dedup_capacity,
    )

    train_out.finalize()
    val_out.finalize()

    return train_wrote, val_wrote


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase3_sft")
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--generic_dataset", type=str, default="teknium/OpenHermes-2.5")
    ap.add_argument("--generic_config", type=str, default=None)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prefetch_buffer", type=int, default=2048)
    ap.add_argument("--encode_batch_size", type=int, default=512)

    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)
    ap.add_argument("--target_train_tokens", type=int, default=75_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=5_000_000)

    ap.add_argument("--val_percent", type=float, default=2.0)

    ap.add_argument("--min_item_tokens", type=int, default=32)
    ap.add_argument("--max_item_tokens", type=int, default=2048)
    ap.add_argument("--dedup_capacity", type=int, default=200_000)

    ap.add_argument("--use_glaive", action="store_true")

    ap.add_argument("--w_generic", type=float, default=0.90)
    ap.add_argument("--w_glaive", type=float, default=0.10)

    args = ap.parse_args()

    hf_token = _hf_token_from_env(args.hf_token)
    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <|endoftext|> token.")
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    print("Phase3 SFT (Single-Pass):")
    print(f"  out_dir: {args.out_dir}")
    print(f"  generic_dataset: {args.generic_dataset} config={args.generic_config}")
    print(f"  use_glaive: {int(args.use_glaive)}")
    print(f"  weights: generic={args.w_generic} glaive={args.w_glaive}")
    print(f"  val_percent: {args.val_percent}")
    print(f"  target_train: {args.target_train_tokens:,} | target_val: {args.target_val_tokens:,}")

    wrote_train, wrote_val = build_both_splits(
        tok=tok,
        eos_id=eos_id,
        out_dir=args.out_dir,
        shard_size=args.bin_shard_size,
        dtype=dtype,
        target_train_tokens=args.target_train_tokens,
        target_val_tokens=args.target_val_tokens,
        seed=args.seed,
        hf_token=hf_token,
        prefetch_buffer=args.prefetch_buffer,
        encode_batch_size=args.encode_batch_size,
        min_item_tokens=args.min_item_tokens,
        max_item_tokens=args.max_item_tokens,
        dedup_capacity=args.dedup_capacity,
        generic_dataset=args.generic_dataset,
        generic_config=args.generic_config,
        val_percent=args.val_percent,
        use_glaive=args.use_glaive,
        w_generic=args.w_generic,
        w_glaive=args.w_glaive,
    )

    print("✅ Phase3 SFT complete:")
    print(f"  train tokens: {wrote_train:,}")
    print(f"  val tokens:   {wrote_val:,}")
    print(f"  out_dir:      {args.out_dir}")


if __name__ == "__main__":
    main()