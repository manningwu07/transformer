#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple

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

    def push_arr(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype, copy=False)
        i = 0
        while i < arr.size:
            space = self.shard_size - self._pos
            take = min(space, arr.size - i)
            self._buf[self._pos : self._pos + take] = arr[i : i + take]
            self._pos += take
            i += take
            if self._pos == self.shard_size:
                self._flush_full()

    def push_ids(self, ids: List[int]) -> None:
        if not ids:
            return
        self.push_arr(np.asarray(ids, dtype=self.dtype))

    def finalize(self) -> None:
        if self._pos > 0:
            self._buf[: self._pos].tofile(self._path())
            self.total_written += int(self._pos)
            self.shard_idx += 1
            self._pos = 0


# ----------------------------
# Helpers
# ----------------------------
def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")


def assign_split_bytes(b: bytes, val_percent: float) -> str:
    h = hashlib.blake2b(b, digest_size=8).digest()
    x = int.from_bytes(h, "little") % 10_000
    return "val" if x < int(val_percent * 100) else "train"


def assign_split_text(s: str, val_percent: float) -> str:
    return assign_split_bytes(s.encode("utf-8"), val_percent)


def format_jsonl_chat(instruction: str, inp: str, out: str) -> str:
    instruction = instruction or ""
    inp = inp or ""
    out = out or ""
    user = f"{instruction}\n\nInput:\n{inp}" if inp.strip() else instruction
    return f"<|user|>\n{user}\n<|assistant|>\n{out}\n"


def format_glaive_v2(ex: dict) -> str:
    """
    Keeps the call/response weaving. Glaive v2 includes:
      - tool call JSON
      - FUNCTION RESPONSE
      - assistant follow-up
    This is the strongest supervised signal you have for "tool weaving".
    """
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


# ----------------------------
# Sources
# ----------------------------
def iter_phase1_token_chunks(phase1_dir: str, pattern: str, dtype: np.dtype, chunk_size: int) -> Iterator[np.ndarray]:
    files = sorted(glob.glob(os.path.join(phase1_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No phase1 shards matching {pattern} in {phase1_dir}")
    for f in files:
        mm = np.memmap(f, dtype=dtype, mode="r")
        n = int(mm.shape[0])
        for start in range(0, n, chunk_size):
            yield np.asarray(mm[start : start + chunk_size])


def iter_synthetic_texts(jsonl_path: str) -> Iterator[str]:
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"synthetic_jsonl not found: {jsonl_path}")
    while True:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                inst = (ex.get("instruction", "") or "").strip()
                inp = (ex.get("input", "") or "").strip()
                out = (ex.get("output", "") or "").strip()
                if inst and out:
                    yield format_jsonl_chat(inst, inp, out)


def iter_glaive_texts(hf_token: Optional[str]) -> Iterator[str]:
    ds = load_dataset(
        "glaiveai/glaive-function-calling-v2",
        split="train",
        streaming=True,
        token=hf_token,
    ).with_format("python")
    for ex in ds:
        s = format_glaive_v2(ex)
        if s:
            yield s


def encode_text_to_ids(tok: Tokenizer, eos_id: int, text: str) -> List[int]:
    ids = tok.encode(text).ids
    if ids and ids[-1] != eos_id:
        ids.append(eos_id)
    return ids


# ----------------------------
# Main write loop (single pass, both splits)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/toolprep_clean")
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--phase1_dir", type=str, required=True)
    ap.add_argument("--phase1_pattern", type=str, default="*train*.bin")
    ap.add_argument("--phase1_dtype", type=str, default="u16", choices=["u16", "u32"])

    ap.add_argument("--synthetic_jsonl", type=str, default="data/raw/synthetic_reasoning_master.jsonl")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_percent", type=float, default=2.0)

    ap.add_argument("--target_train_tokens", type=int, default=450_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=50_000_000)
    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)

    ap.add_argument("--chunk_size", type=int, default=65_536)
    ap.add_argument("--encode_batch_size", type=int, default=256)

    ap.add_argument("--w_tool", type=float, default=0.45)
    ap.add_argument("--w_synth", type=float, default=0.45)
    ap.add_argument("--w_phase1", type=float, default=0.10)

    args = ap.parse_args()

    hf_token = _hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")

    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32
    phase1_dtype = np.uint16 if args.phase1_dtype == "u16" else np.uint32

    # Normalize weights
    weights = [args.w_tool, args.w_synth, args.w_phase1]
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    weights = [w / s for w in weights]

    train_out = BinShardWriterFast(
        out_dir=os.path.join(args.out_dir, "train"),
        prefix="toolprep-train",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )
    val_out = BinShardWriterFast(
        out_dir=os.path.join(args.out_dir, "val"),
        prefix="toolprep-val",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )

    phase1_chunks = iter_phase1_token_chunks(
        args.phase1_dir, args.phase1_pattern, phase1_dtype, args.chunk_size
    )
    synth_texts = iter_synthetic_texts(args.synthetic_jsonl)
    tool_texts = iter_glaive_texts(hf_token)

    rng = random.Random(args.seed)

    train_written = 0
    val_written = 0

    pbar_train = tqdm(total=args.target_train_tokens, unit="tok", desc="toolprep (train)")
    pbar_val = tqdm(total=args.target_val_tokens, unit="tok", desc="toolprep (val)")

    synth_batch: List[str] = []
    tool_batch: List[str] = []

    def flush_text_batch(batch: List[str]) -> List[List[int]]:
        if not batch:
            return []
        encs = tok.encode_batch(batch)
        out_ids = []
        for enc in encs:
            ids = enc.ids
            if ids and ids[-1] != eos_id:
                ids.append(eos_id)
            if ids:
                out_ids.append(ids)
        return out_ids

    while train_written < args.target_train_tokens or val_written < args.target_val_tokens:
        # choose a source that can still produce data
        choice = rng.choices([0, 1, 2], weights=weights, k=1)[0]

        if choice == 2:
            # phase1 chunk is already tokenized
            try:
                chunk = next(phase1_chunks)
            except StopIteration:
                weights[2] = 0.0
                continue

            b = chunk.tobytes()
            split = assign_split_bytes(b, args.val_percent)

            if split == "val" and val_written < args.target_val_tokens:
                val_out.push_arr(chunk.astype(dtype, copy=False))
                val_written += int(chunk.size)
                pbar_val.update(int(chunk.size))
            elif split == "train" and train_written < args.target_train_tokens:
                train_out.push_arr(chunk.astype(dtype, copy=False))
                train_written += int(chunk.size)
                pbar_train.update(int(chunk.size))
            else:
                # saturating: if preferred split is full, route to other if possible
                if val_written < args.target_val_tokens:
                    val_out.push_arr(chunk.astype(dtype, copy=False))
                    val_written += int(chunk.size)
                    pbar_val.update(int(chunk.size))
                elif train_written < args.target_train_tokens:
                    train_out.push_arr(chunk.astype(dtype, copy=False))
                    train_written += int(chunk.size)
                    pbar_train.update(int(chunk.size))
            continue

        if choice == 1:
            # synthetic text
            t = next(synth_texts).strip()
            if not t:
                continue
            synth_batch.append(t)
            if len(synth_batch) < args.encode_batch_size:
                continue
            items = flush_text_batch(synth_batch)
            synth_batch = []
        else:
            # tool text
            t = next(tool_texts).strip()
            if not t:
                continue
            tool_batch.append(t)
            if len(tool_batch) < args.encode_batch_size:
                continue
            items = flush_text_batch(tool_batch)
            tool_batch = []

        for ids in items:
            if train_written >= args.target_train_tokens and val_written >= args.target_val_tokens:
                break

            text_for_hash = tok.decode(ids[: min(len(ids), 2048)])
            split = assign_split_text(text_for_hash, args.val_percent)

            if split == "val" and val_written < args.target_val_tokens:
                val_out.push_ids(ids)
                val_written += len(ids)
                pbar_val.update(len(ids))
            elif split == "train" and train_written < args.target_train_tokens:
                train_out.push_ids(ids)
                train_written += len(ids)
                pbar_train.update(len(ids))
            else:
                # saturating routing
                if val_written < args.target_val_tokens:
                    val_out.push_ids(ids)
                    val_written += len(ids)
                    pbar_val.update(len(ids))
                elif train_written < args.target_train_tokens:
                    train_out.push_ids(ids)
                    train_written += len(ids)
                    pbar_train.update(len(ids))

    # flush any remaining text batches
    for batch, is_tool in [(tool_batch, True), (synth_batch, False)]:
        items = flush_text_batch(batch)
        for ids in items:
            if train_written >= args.target_train_tokens and val_written >= args.target_val_tokens:
                break
            text_for_hash = tok.decode(ids[: min(len(ids), 2048)])
            split = assign_split_text(text_for_hash, args.val_percent)
            if split == "val" and val_written < args.target_val_tokens:
                val_out.push_ids(ids)
                val_written += len(ids)
                pbar_val.update(len(ids))
            elif train_written < args.target_train_tokens:
                train_out.push_ids(ids)
                train_written += len(ids)
                pbar_train.update(len(ids))

    train_out.finalize()
    val_out.finalize()
    pbar_train.close()
    pbar_val.close()

    print("✅ toolprep_clean complete:")
    print(f"  train tokens: {train_written:,}")
    print(f"  val tokens:   {val_written:,}")
    print(f"  out_dir:      {args.out_dir}")


if __name__ == "__main__":
    main()