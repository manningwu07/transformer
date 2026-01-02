#!/usr/bin/env python3
import argparse
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


def load_dataset_streaming(
    name: str,
    config: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
):
    """
    Streaming loader compatible with different datasets versions:
    tries token= first, falls back to use_auth_token=.
    """
    kwargs = {"split": split, "streaming": True}
    args = (name, config) if config is not None else (name,)

    if token:
        try:
            return load_dataset(*args, token=token, **kwargs)
        except TypeError:
            return load_dataset(*args, use_auth_token=token, **kwargs)

    return load_dataset(*args, **kwargs)


def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )


# -------- Formatting --------
def format_jsonl_chat(instruction: str, inp: str, out: str) -> str:
    """
    Your synthetic JSONL -> ChatML
    """
    instruction = instruction or ""
    inp = inp or ""
    out = out or ""
    user = f"{instruction}\n\nInput:\n{inp}" if inp.strip() else instruction
    return f"<|user|>\n{user}\n<|assistant|>\n{out}\n"


def format_glaive_v2(ex: dict) -> str:
    """
    Converts Glaive v2 'chat' into ChatML + your custom tool tags.
    """
    text = (ex.get("chat", "") or "").strip()
    if not text:
        return ""

    # Standardize ChatML labels
    text = text.replace("SYSTEM: ", "<|system|>\n")
    text = text.replace("USER: ", "\n<|user|>\n")
    text = text.replace("ASSISTANT: ", "\n<|assistant|>\n")

    # Remove dataset EOS; we append EOS later
    text = text.replace("<|endoftext|>", "")

    # Wrap function responses
    text = re.sub(
        r"FUNCTION RESPONSE:\s*(.*?)(?=\n<\|user\|>|\n<\|assistant\|>|\n<\|system\|>|\Z)",
        r"<response>\1</response>",
        text,
        flags=re.DOTALL,
    )

    # Convert OpenAI-ish tool call JSON blobs into tags
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


def pick_first_present(ex: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    return None


# -------- Interleave without HF schema inference --------
def probabilistic_interleave(
    sources: list[Iterator[str]],
    probs: list[float],
    seed: int,
) -> Iterator[str]:
    assert len(sources) == len(probs)
    rng = random.Random(seed)
    active = [True] * len(sources)
    weights = probs[:]

    while any(active):
        idx = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        if not active[idx] or weights[idx] <= 0.0:
            continue
        try:
            yield next(sources[idx])
        except StopIteration:
            active[idx] = False
            weights[idx] = 0.0


# -------- Sharding --------
@dataclass
class ShardState:
    split: str
    out_dir: str
    shard_size: int
    dtype: np.dtype
    shard_idx: int = 0
    total_written: int = 0
    buf: List[int] = None

    def __post_init__(self):
        self.buf = []
        os.makedirs(self.out_dir, exist_ok=True)

    def _shard_path(self) -> str:
        return os.path.join(
            self.out_dir,
            f"phase2-{self.split}-{self.shard_idx:05d}.bin",
        )

    def push(self, token_ids: List[int]) -> None:
        self.buf.extend(token_ids)
        self._flush_full_shards()

    def _flush_full_shards(self) -> None:
        while len(self.buf) >= self.shard_size:
            chunk = self.buf[: self.shard_size]
            arr = np.asarray(chunk, dtype=self.dtype)
            arr.tofile(self._shard_path())
            self.total_written += int(arr.size)
            self.shard_idx += 1
            self.buf = self.buf[self.shard_size :]

    def finalize(self) -> None:
        if self.buf:
            arr = np.asarray(self.buf, dtype=self.dtype)
            arr.tofile(self._shard_path())
            self.total_written += int(arr.size)
            self.shard_idx += 1
            self.buf = []


def encode_stream_to_shards(
    text_iter: Iterator[str],
    tokenizer: Tokenizer,
    train_state: ShardState,
    val_state: Optional[ShardState],
    target_tokens: int,
    eos_id: int,
    seed: int,
    val_fraction: float,
    encode_batch_size: int,
):
    rng = random.Random(seed)
    pbar = tqdm(total=target_tokens, unit="tok", desc="Phase2 Sharding")
    written_so_far = 0
    batch: list[str] = []

    def flush_batch(batch_texts: list[str]) -> None:
        nonlocal written_so_far
        encs = tokenizer.encode_batch(batch_texts)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)

            is_val = rng.random() < val_fraction if val_state else False
            target = val_state if is_val else train_state
            target.push(ids)

            written_so_far += len(ids)
            pbar.update(len(ids))
            if written_so_far >= target_tokens:
                return

    for t in text_iter:
        t = (t or "").strip()
        if not t:
            continue
        batch.append(t)
        if len(batch) >= encode_batch_size:
            flush_batch(batch)
            batch = []
            if written_so_far >= target_tokens:
                break

    if batch and written_so_far < target_tokens:
        flush_batch(batch)

    pbar.close()
    return train_state.total_written, (val_state.total_written if val_state else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase2")
    ap.add_argument("--target_tokens", type=int, default=5_000_000_000)
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--val_fraction", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--encode_batch_size", type=int, default=1024)

    # Your synthetic planning JSONL
    ap.add_argument(
        "--synthetic_jsonl",
        type=str,
        default="data/raw/synthetic_reasoning_master.jsonl",
    )

    ap.add_argument("--hf_token", type=str, default=None)

    # Stack-Edu MUST have config(s)
    ap.add_argument(
        "--stack_edu_configs",
        type=str,
        default="Python,Cpp,Go,JavaScript,Rust",
        help=(
            "Comma-separated stack-edu configs. Valid: "
            "C,CSharp,Cpp,Go,Java,JavaScript,Markdown,PHP,Python,Ruby,Rust,"
            "SQL,Shell,Swift,TypeScript"
        ),
    )

    args = ap.parse_args()
    hf_token = _hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|> token.")
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    train_state = ShardState(
        "train",
        os.path.join(args.out_dir, "train"),
        args.shard_size,
        dtype,
    )
    val_state = ShardState(
        "val",
        os.path.join(args.out_dir, "val"),
        args.shard_size,
        dtype,
    )

    print("Loading Phase 2 sources...")

    # 1) Synthetic planning
    syn_ds = load_dataset(
        "json",
        data_files=args.synthetic_jsonl,
        split="train",
        streaming=True,
    )

    def syn_iter() -> Iterator[str]:
        for ex in syn_ds:
            inst = ex.get("instruction", "") or ""
            inp = ex.get("input", "") or ""
            out = ex.get("output", "") or ""
            if not inst.strip() or not out.strip():
                continue
            yield format_jsonl_chat(inst, inp, out)

    # 2) Tools (Glaive v2)
    glaive_ds = load_dataset_streaming(
        "glaiveai/glaive-function-calling-v2",
        split="train",
        token=hf_token,
    )

    def glaive_iter() -> Iterator[str]:
        for ex in glaive_ds:
            chat = ex.get("chat", "")
            if isinstance(chat, str) and chat.strip():
                yield format_glaive_v2(ex)

    # 3) Math (FineMath)
    fm_ds = load_dataset_streaming(
        "HuggingFaceTB/finemath",
        config="finemath-3plus",
        split="train",
        token=hf_token,
    )

    def fm_iter() -> Iterator[str]:
        for ex in fm_ds:
            t = ex.get("text", "")
            if isinstance(t, str) and t.strip():
                yield t

    # 4) Code (Stack-Edu) — load MULTIPLE configs (languages)
    cfgs = [c.strip() for c in args.stack_edu_configs.split(",") if c.strip()]
    if not cfgs:
        raise ValueError("Empty --stack_edu_configs (stack-edu requires configs).")

    code_iters: list[Iterator[str]] = []
    for cfg in cfgs:
        ds = load_dataset_streaming(
            "HuggingFaceTB/stack-edu",
            config=cfg,
            split="train",
            token=hf_token,
        )

        def _make_iter(dset):
            def _it():
                for ex in dset:
                    t = pick_first_present(ex, ["content", "text", "code"])
                    if t:
                        yield t

            return _it()

        code_iters.append(_make_iter(ds))

    def code_mix() -> Iterator[str]:
        return probabilistic_interleave(
            sources=code_iters,
            probs=[1.0 / len(code_iters)] * len(code_iters),
            seed=args.seed + 999,
        )

    # Final mix: 5% planning, 5% tools, 50% math, 40% code
    text_generator = probabilistic_interleave(
        sources=[syn_iter(), glaive_iter(), fm_iter(), code_mix()],
        probs=[0.05, 0.05, 0.50, 0.40],
        seed=args.seed,
    )

    train_written, val_written = encode_stream_to_shards(
        text_generator,
        tok,
        train_state,
        val_state,
        args.target_tokens,
        eos_id,
        args.seed,
        args.val_fraction,
        args.encode_batch_size,
    )

    train_state.finalize()
    val_state.finalize()

    print(
        f"✅ Phase 2 complete. train={train_written:,} tok, val={val_written:,} tok"
    )
    print(f"Shards at: {args.out_dir}")


if __name__ == "__main__":
    main()