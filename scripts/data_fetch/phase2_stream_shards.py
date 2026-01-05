#!/usr/bin/env python3
# scripts/phase2_stream_shards.py (FIXED)
import argparse
import json
import os
import random
import re
import signal
import sys
from dataclasses import dataclass
from typing import Iterator, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

import time
import queue
import threading

# -------- Prefetch (prevents HF streaming stalls from freezing the whole run) --------
class PrefetchIter:
    def __init__(
        self,
        name: str,
        it: Iterator[str],
        max_buffer: int = 64,
    ):
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
                # Block here, not in main thread.
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
            # Surface the error once; after that treat as exhausted.
            e = self._err
            self._err = None
            raise e
        try:
            item = self._q.get_nowait()
        except queue.Empty:
            return None
        return item

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
        # Only choose among sources that currently have buffered items.
        available = []
        weights = []
        for i, s in enumerate(sources):
            if s._q.empty():
                continue
            if probs[i] <= 0:
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

def timeout_handler(signum, frame):
    raise TimeoutError("Dataset fetch timed out")


def load_dataset_with_timeout(name, config=None, split="train", token=None, timeout_sec=120):
    """Load streaming dataset with a timeout to prevent indefinite hangs."""
    kwargs = {"split": split, "streaming": True}
    args = (name, config) if config else (name,)
    
    if token:
        kwargs["token"] = token
    
    print(f"  → Loading {name}" + (f" [{config}]" if config else "") + "...", flush=True)
    
    ds = load_dataset(*args, **kwargs)
    
    # Force the first item to verify connectivity
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        it = iter(ds)
        first = next(it)
        print(f"    ✓ Connected to {name}", flush=True)
        # Return a generator that yields the first item, then the rest
        def _gen():
            yield first
            yield from it
        return _gen()
    except TimeoutError:
        print(f"    ✗ Timeout on {name}, skipping", flush=True)
        return iter([])  # Empty iterator
    except Exception as e:
        print(f"    ✗ Error on {name}: {e}", flush=True)
        return iter([])
    finally:
        signal.alarm(0)


def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


# -------- Formatting --------
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


# -------- FIXED Interleave --------
def probabilistic_interleave(
    sources: List[Iterator[str]],
    probs: List[float],
    seed: int,
) -> Iterator[str]:
    """
    Probabilistically interleave multiple iterators.
    FIXED: Properly handles depleted sources without spinning.
    """
    assert len(sources) == len(probs)
    rng = random.Random(seed)
    
    # Wrap iterators to track exhaustion + buffer one item
    buffers = [None] * len(sources)
    active = [True] * len(sources)
    
    # Prime buffers
    for i, src in enumerate(sources):
        try:
            buffers[i] = next(src)
        except StopIteration:
            active[i] = False
    
    while any(active):
        # Build weights only for active sources
        weights = [probs[i] if active[i] else 0.0 for i in range(len(sources))]
        total = sum(weights)
        if total <= 0:
            break
        
        # Normalize
        weights = [w / total for w in weights]
        
        idx = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        
        # Yield buffered item
        yield buffers[idx]
        
        # Refill buffer
        try:
            buffers[idx] = next(sources[idx])
        except StopIteration:
            active[idx] = False
            buffers[idx] = None


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
            chunk = self.buf[:self.shard_size]
            arr = np.asarray(chunk, dtype=self.dtype)
            arr.tofile(self._shard_path())
            self.total_written += int(arr.size)
            self.shard_idx += 1
            self.buf = self.buf[self.shard_size:]
            print(f"  💾 Wrote {self.split} shard {self.shard_idx - 1}", flush=True)

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
    val_state: ShardState,
    target_train_tokens: int,
    target_val_tokens: int,
    eos_id: int,
    seed: int,
    flush_interval_sec: float,
    encode_batch_size: int,
):
    rng = random.Random(seed)
    total_target = int(target_train_tokens + target_val_tokens)
    pbar = tqdm(total=total_target, unit="tok", desc="Phase2 Sharding")
    train_written = 0
    val_written = 0
    batch: List[str] = []
    last_flush_t = time.time()
    

    def flush_batch(batch_texts: List[str]) -> int:
        nonlocal train_written, val_written, last_flush_t
        encs = tokenizer.encode_batch(batch_texts)
        batch_tokens = 0
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)
            
            # Exact token-budget routing (not probabilistic drift)
            remaining_train = max(0, target_train_tokens - train_written)
            remaining_val = max(0, target_val_tokens - val_written)
            if remaining_train <= 0 and remaining_val <= 0:
                return

            # Prefer whichever split has more remaining budget (softly randomized)
            choose_val_p = (
                remaining_val / max(1, remaining_train + remaining_val)
                if val_state is not None
                else 0.0
            )
            to_val = val_state is not None and remaining_val > 0 and rng.random() < choose_val_p

            if to_val:
                val_state.push(ids)
                val_written += len(ids)
            else:
                train_state.push(ids)
                train_written += len(ids)

            pbar.update(len(ids))
            if train_written >= target_train_tokens and val_written >= target_val_tokens:
                return
        last_flush_t = time.time()

    stall_count = 0
    for t in text_iter:
        t = (t or "").strip()
        if not t:
            stall_count += 1
            if stall_count > 10000:
                print("⚠️ 10k empty docs in a row, possible data issue", flush=True)
                stall_count = 0
            continue
        stall_count = 0
        
        batch.append(t)
        now = time.time()
        if len(batch) >= encode_batch_size or (now - last_flush_t) >= flush_interval_sec:
            flush_batch(batch)
            batch = []
            if train_written >= target_train_tokens and val_written >= target_val_tokens:
                break

    if batch and (train_written < target_train_tokens or val_written < target_val_tokens):
        flush_batch(batch)

    pbar.close()
    return train_state.total_written, (val_state.total_written if val_state else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase2")
    ap.add_argument("--target_train_tokens", type=int, default=4_500_000_000)
    ap.add_argument("--target_val_tokens", type=int, default=500_000_000)
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--encode_batch_size", type=int, default=512)
    ap.add_argument("--synthetic_jsonl", type=str, default="data/raw/synthetic_reasoning_master.jsonl")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--stack_edu_configs", type=str, default="Python,TypeScript,Go,Cpp,Rust")
    ap.add_argument("--flush_interval_sec", type=float, default=2.0)
    
    # Debugging flags
    ap.add_argument("--skip_hf", action="store_true", help="Skip HF datasets, only use local JSONL")
    
    args = ap.parse_args()
    hf_token = _hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer missing <|endoftext|>")
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    train_state = ShardState("train", os.path.join(args.out_dir, "train"), args.shard_size, dtype)
    val_state = ShardState("val", os.path.join(args.out_dir, "val"), args.shard_size, dtype)

    print("=" * 60)
    print("🚀 Phase 2 Sharding - Loading Sources")
    print("=" * 60, flush=True)

    sources = []
    probs = []

    # 1) Synthetic JSONL (local, fast)
    if os.path.exists(args.synthetic_jsonl):
        print(f"📄 Loading synthetic JSONL: {args.synthetic_jsonl}", flush=True)
        syn_ds = load_dataset("json", data_files=args.synthetic_jsonl, split="train", streaming=True)
        
        def syn_iter():
            for ex in syn_ds:
                inst = ex.get("instruction", "") or ""
                out = ex.get("output", "") or ""
                if inst.strip() and out.strip():
                    yield format_jsonl_chat(inst, ex.get("input", ""), out)
        
        sources.append(syn_iter())
        probs.append(0.05)
        print("  ✓ Synthetic loaded", flush=True)
    else:
        print(f"⚠️ Synthetic JSONL not found: {args.synthetic_jsonl}", flush=True)

    if not args.skip_hf:
        # 2) Glaive (tool calling)
        glaive_iter = load_dataset_with_timeout(
            "glaiveai/glaive-function-calling-v2",
            split="train",
            token=hf_token,
        )
        
        def glaive_gen():
            for ex in glaive_iter:
                chat = ex.get("chat", "")
                if isinstance(chat, str) and chat.strip():
                    formatted = format_glaive_v2(ex)
                    if formatted:
                        yield formatted
        
        sources.append(glaive_gen())
        probs.append(0.05)

        # 3) FineMath
        fm_iter = load_dataset_with_timeout(
            "HuggingFaceTB/finemath",
            config="finemath-3plus",
            split="train",
            token=hf_token,
        )
        
        def fm_gen():
            for ex in fm_iter:
                t = ex.get("text", "")
                if isinstance(t, str) and t.strip():
                    yield t
        
        sources.append(fm_gen())
        probs.append(0.50)

        # 4) Stack-Edu (multiple languages)
        cfgs = [c.strip() for c in args.stack_edu_configs.split(",") if c.strip()]
        code_sources = []
        
        for cfg in cfgs:
            code_iter = load_dataset_with_timeout(
                "HuggingFaceTB/stack-edu",
                config=cfg,
                split="train",
                token=hf_token,
            )
            
            def make_code_gen(it):
                def _gen():
                    for ex in it:
                        for key in ["content", "text", "code"]:
                            t = ex.get(key, "")
                            if isinstance(t, str) and t.strip():
                                yield t
                                break
                return _gen()
            
            code_sources.append(make_code_gen(code_iter))
        
        if code_sources:
            code_probs = [1.0 / len(code_sources)] * len(code_sources)
            code_mix = probabilistic_interleave(code_sources, code_probs, args.seed + 999)
            sources.append(code_mix)
            probs.append(0.40)

    if not sources:
        raise RuntimeError("No data sources available!")

    # Normalize probs
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]
    
    print(f"\n📊 Source distribution: {probs}")
    print("=" * 60, flush=True)

    raw_sources = [
        ("synthetic", syn_iter()),
        ("glaive", glaive_gen()),
        ("finemath", fm_gen()),
        ("code", code_mix),
    ]
    prefetch_sources = [
        PrefetchIter(name=n, it=it, max_buffer=64) for (n, it) in raw_sources
    ]
    probs = [0.05, 0.05, 0.50, 0.40]
    text_generator = interleave_nonblocking(
        sources=prefetch_sources,
        probs=probs,
        seed=args.seed,
    )

    train_written, val_written = encode_stream_to_shards(
        text_generator,
        tok,
        train_state,
        val_state,
        args.target_train_tokens,
        args.target_val_tokens,
        eos_id,
        args.seed,
        args.encode_batch_size,
        args.flush_interval_sec,
     )

    train_state.finalize()
    val_state.finalize()

    print("=" * 60)
    print(f"✅ Phase 2 complete!")
    print(f"   Train: {train_written:,} tokens ({train_state.shard_idx} shards)")
    print(f"   Val:   {val_written:,} tokens ({val_state.shard_idx} shards)")
    print(f"   Output: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()