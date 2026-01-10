#!/usr/bin/env python3
# Goes between phase 1 and phase 2 to build tool-synth-prep shards.
"""
Build Tool-Prep shards: (tool calls + synthetic planning + phase1 replay).

Goal:
- Clear signal for tool calling + planning formatting
- Still replay some phase1 tokens to prevent forgetting

Mix (by tokens, approx via fixed chunking):
  tool (glaive) : synth : phase1 = 40 : 40 : 20

Output:
  out_dir/{train,val}/*.bin
"""

import argparse
import glob
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Iterator, Optional, List

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


# -------------------- Formatting --------------------
def format_jsonl_chat(instruction: str, inp: str, out: str) -> str:
    instruction = instruction or ""
    inp = inp or ""
    out = out or ""
    user = f"{instruction}\n\nInput:\n{inp}" if inp.strip() else instruction
    return f"<|user|>\n{user}\n<|assistant|>\n{out}\n"


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


def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


# -------------------- Sharding --------------------
@dataclass
class ShardWriter:
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

    def push(self, tokens: np.ndarray) -> None:
        if tokens.size == 0:
            return
        self.buf.extend(tokens.tolist())
        self._flush()

    def _flush(self) -> None:
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


# -------------------- Token Sources --------------------
def iter_phase1_tokens(
    shard_dir: str,
    pattern: str,
    dtype: np.dtype,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    files = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No phase1 shards matching {pattern} in {shard_dir}")

    for f in files:
        mm = np.memmap(f, dtype=dtype, mode="r")
        for start in range(0, len(mm), chunk_size):
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
                inst = ex.get("instruction", "") or ""
                inp = ex.get("input", "") or ""
                out = ex.get("output", "") or ""
                if inst.strip() and out.strip():
                    yield format_jsonl_chat(inst, inp, out)


def iter_glaive_texts(hf_token: Optional[str]) -> Iterator[str]:
    ds = load_dataset(
        "glaiveai/glaive-function-calling-v2",
        split="train",
        streaming=True,
        token=hf_token,
    )
    for ex in ds:
        formatted = format_glaive_v2(ex)
        if formatted:
            yield formatted


class EncodedChunkStream:
    """
    Turns an iterator of texts into fixed-size token chunks.
    This makes interleaving probabilities approximately token-proportional.
    """

    def __init__(
        self,
        text_iter: Iterator[str],
        tokenizer: Tokenizer,
        eos_id: int,
        chunk_size: int,
        encode_batch_size: int,
    ):
        self.text_iter = text_iter
        self.tok = tokenizer
        self.eos_id = int(eos_id)
        self.chunk_size = int(chunk_size)
        self.encode_batch_size = int(encode_batch_size)
        self.buf: List[int] = []

    def __iter__(self) -> "EncodedChunkStream":
        return self

    def __next__(self) -> np.ndarray:
        while len(self.buf) < self.chunk_size:
            batch: List[str] = []
            try:
                for _ in range(self.encode_batch_size):
                    t = next(self.text_iter)
                    t = (t or "").strip()
                    if t:
                        batch.append(t)
            except StopIteration:
                pass

            if not batch and len(self.buf) == 0:
                raise StopIteration

            if batch:
                encs = self.tok.encode_batch(batch)
                for enc in encs:
                    ids = enc.ids
                    if not ids:
                        continue
                    if ids[-1] != self.eos_id:
                        ids.append(self.eos_id)
                    self.buf.extend(ids)

            if not batch and len(self.buf) > 0:
                break

        if len(self.buf) == 0:
            raise StopIteration

        out = self.buf[: self.chunk_size]
        self.buf = self.buf[self.chunk_size :]
        return np.asarray(out, dtype=np.int64)


def interleave_token_streams(
    sources: List[Iterator[np.ndarray]],
    weights: List[float],
    seed: int,
) -> Iterator[np.ndarray]:
    rng = random.Random(seed)
    active = [True] * len(sources)
    w = [float(x) for x in weights]

    while any(active):
        total = sum(w[i] for i in range(len(w)) if active[i] and w[i] > 0)
        if total <= 0:
            break

        probs = [w[i] / total if active[i] else 0.0 for i in range(len(w))]
        idx = rng.choices(range(len(sources)), weights=probs, k=1)[0]

        if not active[idx]:
            continue

        try:
            yield next(sources[idx])
        except StopIteration:
            active[idx] = False
            w[idx] = 0.0


def build_split(
    *,
    split: str,
    out_dir: str,
    tokenizer: Tokenizer,
    eos_id: int,
    dtype: np.dtype,
    shard_size: int,
    target_tokens: int,
    seed: int,
    phase1_dir: str,
    phase1_pattern: str,
    phase1_dtype: np.dtype,
    synthetic_jsonl: str,
    hf_token: Optional[str],
    weights: List[float],
    chunk_size: int,
    encode_batch_size: int,
) -> None:
    writer = ShardWriter(
        out_dir=os.path.join(out_dir, split),
        prefix=f"toolprep-{split}",
        shard_size=shard_size,
        dtype=dtype,
    )

    # 1) Phase1 (already tokenized)
    phase1_stream = iter_phase1_tokens(
        shard_dir=phase1_dir,
        pattern=phase1_pattern,
        dtype=phase1_dtype,
        chunk_size=chunk_size,
    )

    # 2) Synthetic (looping)
    synth_stream = EncodedChunkStream(
        text_iter=iter_synthetic_texts(synthetic_jsonl),
        tokenizer=tokenizer,
        eos_id=eos_id,
        chunk_size=chunk_size,
        encode_batch_size=encode_batch_size,
    )

    # 3) Tool calling (Glaive)
    tool_stream = EncodedChunkStream(
        text_iter=iter_glaive_texts(hf_token),
        tokenizer=tokenizer,
        eos_id=eos_id,
        chunk_size=chunk_size,
        encode_batch_size=encode_batch_size,
    )

    streams = [tool_stream, synth_stream, phase1_stream]

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"toolprep:{split}")
    for chunk in interleave_token_streams(streams, weights, seed=seed):
        writer.push(chunk.astype(dtype, copy=False))
        pbar.update(int(chunk.size))
        if writer.total_written >= target_tokens:
            break

    writer.finalize()
    pbar.close()
    print(
        f"✅ {split}: wrote {writer.total_written:,} tokens "
        f"({writer.shard_idx} shards) -> {os.path.join(out_dir, split)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/toolprep")

    ap.add_argument("--phase1_dir", type=str, required=True)
    ap.add_argument("--phase1_pattern", type=str, default="*train*.bin")
    ap.add_argument(
        "--phase1_dtype",
        type=str,
        default="u16",
        choices=["u16", "u32"],
        help="dtype of phase1 shards",
    )

    ap.add_argument(
        "--synthetic_jsonl",
        type=str,
        default="data/raw/synthetic_reasoning_master.jsonl",
    )
    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--target_train_tokens", type=int, default=450_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=50_000_000)
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--weights",
        type=str,
        default="0.40,0.40,0.20",
        help="tool,synth,phase1",
    )
    ap.add_argument("--chunk_size", type=int, default=65_536)
    ap.add_argument("--encode_batch_size", type=int, default=256)

    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")

    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32
    phase1_dtype = np.uint16 if args.phase1_dtype == "u16" else np.uint32

    weights = [float(x) for x in args.weights.split(",")]
    if len(weights) != 3:
        raise ValueError("--weights must have 3 values: tool,synth,phase1")
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    weights = [w / s for w in weights]

    hf_token = _hf_token_from_env(args.hf_token)

    print(f"Mix weights (tool,synth,phase1): {weights}")
    print(f"Out: {args.out_dir}")

    build_split(
        split="train",
        out_dir=args.out_dir,
        tokenizer=tok,
        eos_id=eos_id,
        dtype=dtype,
        shard_size=args.shard_size,
        target_tokens=args.target_train_tokens,
        seed=args.seed,
        phase1_dir=args.phase1_dir,
        phase1_pattern=args.phase1_pattern,
        phase1_dtype=phase1_dtype,
        synthetic_jsonl=args.synthetic_jsonl,
        hf_token=hf_token,
        weights=weights,
        chunk_size=args.chunk_size,
        encode_batch_size=args.encode_batch_size,
    )

    build_split(
        split="val",
        out_dir=args.out_dir,
        tokenizer=tok,
        eos_id=eos_id,
        dtype=dtype,
        shard_size=args.shard_size,
        target_tokens=args.target_val_tokens,
        seed=args.seed + 999,
        phase1_dir=args.phase1_dir,
        phase1_pattern=args.phase1_pattern,
        phase1_dtype=phase1_dtype,
        synthetic_jsonl=args.synthetic_jsonl,
        hf_token=hf_token,
        weights=weights,
        chunk_size=args.chunk_size,
        encode_batch_size=args.encode_batch_size,
    )


if __name__ == "__main__":
    main()