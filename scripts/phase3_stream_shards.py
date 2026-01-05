#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import time
import queue
import threading
import hashlib

class PrefetchIter:
    """
    Pull from a (possibly-stalling) iterator in a background thread.
    Main thread never blocks on HF streaming.
    """

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
            # only choose sources that currently have buffered items
            if s._q.empty():
                continue
            if probs[i] <= 0.0:
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


def load_dataset_streaming(
    name: str,
    config: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
):
    """
    Streaming loader that works across HF datasets versions.
    Forces python format to avoid Arrow schema inference hangs.
    """
    kwargs = {"split": split, "streaming": True}
      # If we have a token, also export it so HF Hub resolver uses authenticated
    # requests (avoids "unauthenticated requests" warning + rate limits).
    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    if config is not None:
        args = (name, config)
    else:
        args = (name,)

    if token:
        try:
            ds = load_dataset(*args, token=token, **kwargs)
        except TypeError:
            ds = load_dataset(*args, use_auth_token=token, **kwargs)
    else:
        ds = load_dataset(*args, **kwargs)

    # Critical: avoid Arrow iteration/schema inference that can hang
    return ds.with_format("python")


def _hf_token_from_env(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )


def format_chatml(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n"


def format_synth_chat(ex: dict) -> Optional[str]:
    """
    Your synthetic JSONL:
      {"instruction": "...", "input": "...", "output": "..."}
    -> ChatML
    """
    inst = (ex.get("instruction", "") or "").strip()
    inp = (ex.get("input", "") or "").strip()
    out = (ex.get("output", "") or "").strip()
    if not inst or not out:
        return None
    user = f"{inst}\n\nInput:\n{inp}" if inp else inst
    return format_chatml(user, out)


def pick_first_present(ex: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    return None


def probabilistic_interleave(
    sources: list[Iterator[str]],
    probs: list[float],
    seed: int,
) -> Iterator[str]:
    """
    Interleave arbitrary generators by probability.
    Removes sources as they exhaust.
    """
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


@dataclass
class BinShardWriter:
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

    def push(self, ids: List[int]) -> None:
        self.buf.extend(ids)
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


@dataclass
class DpoShardWriter:
    out_dir: str
    prefix: str
    shard_examples: int
    dtype: np.dtype
    shard_idx: int = 0
    n_written: int = 0

    prompt_tokens: List[int] = None
    chosen_tokens: List[int] = None
    rejected_tokens: List[int] = None
    prompt_offsets: List[int] = None
    chosen_offsets: List[int] = None
    rejected_offsets: List[int] = None

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._reset()

    def _reset(self):
        self.prompt_tokens = []
        self.chosen_tokens = []
        self.rejected_tokens = []
        self.prompt_offsets = [0]
        self.chosen_offsets = [0]
        self.rejected_offsets = [0]

    def _path(self) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:05d}.npz")

    def push(self, prompt: List[int], chosen: List[int], rejected: List[int]) -> None:
        self.prompt_tokens.extend(prompt)
        self.chosen_tokens.extend(chosen)
        self.rejected_tokens.extend(rejected)
        self.prompt_offsets.append(len(self.prompt_tokens))
        self.chosen_offsets.append(len(self.chosen_tokens))
        self.rejected_offsets.append(len(self.rejected_tokens))

        self.n_written += 1
        if self.n_written >= self.shard_examples:
            self.flush()

    def flush(self) -> None:
        if self.n_written == 0:
            return
        np.savez(
            self._path(),
            prompt=np.asarray(self.prompt_tokens, dtype=self.dtype),
            chosen=np.asarray(self.chosen_tokens, dtype=self.dtype),
            rejected=np.asarray(self.rejected_tokens, dtype=self.dtype),
            prompt_offsets=np.asarray(self.prompt_offsets, dtype=np.int64),
            chosen_offsets=np.asarray(self.chosen_offsets, dtype=np.int64),
            rejected_offsets=np.asarray(self.rejected_offsets, dtype=np.int64),
        )
        self.shard_idx += 1
        self.n_written = 0
        self._reset()

    def finalize(self) -> None:
        self.flush()


def encode_text_stream_to_bin(
    text_iter: Iterator[str],
    tok: Tokenizer,
    eos_id: int,
    out: BinShardWriter,
    target_tokens: int,
    encode_batch_size: int,
    flush_interval_sec: float,
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
            out.push(ids)
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


def get_fineweb_score(ex: dict) -> Optional[float]:
    """
    fineweb-edu-dedup sometimes has metadata as dict; be defensive.
    """
    md = ex.get("metadata", None)
    if isinstance(md, dict):
        sc = md.get("score", None)
        if isinstance(sc, (int, float)):
            return float(sc)
        return None
    if isinstance(md, str):
        try:
            obj = json.loads(md)
            sc = obj.get("score", None)
            if isinstance(sc, (int, float)):
                return float(sc)
        except Exception:
            return None
    return None


def degrade_answer(answer: str, seed: int) -> str:
    """
    Deterministic "worse answer" for DPO rejected samples.
    """
    rng = random.Random(seed)
    a = (answer or "").strip()
    if not a:
        return "I don't know."

    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    if len(lines) >= 3:
        rng.shuffle(lines)
        lines = lines[: max(1, len(lines) // 2)]
        a2 = "\n".join(lines)
    else:
        a2 = a[: max(1, len(a) // 3)]

    return f"{a2}\n\n(Note: This plan might be incorrect.)"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase3")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--encode_batch_size", type=int, default=16)
    ap.add_argument("--flush_interval_sec", type=float, default=2.0)
    ap.add_argument("--prefetch_buffer", type=int, default=64)
    ap.add_argument("--bin_shard_size", type=int, default=100_000_000)

    ap.add_argument("--target_tokens_longctx", type=int, default=4_000_000_000)
    ap.add_argument("--target_tokens_sft", type=int, default=1_000_000_000)
    ap.add_argument("--target_dpo_pairs", type=int, default=500_000)
    ap.add_argument("--dpo_shard_examples", type=int, default=20_000)

    ap.add_argument(
        "--synthetic_jsonl",
        type=str,
        default="data/raw/synthetic_reasoning_master.jsonl",
    )

    # Reasoning dataset switches
    ap.add_argument("--use_smollm_corpus", action="store_true")
    ap.add_argument("--use_finemath", action="store_true")
    ap.add_argument("--use_stack_edu", action="store_true")
    ap.add_argument("--use_gsm8k", action="store_true")
    ap.add_argument("--use_apps", action="store_true")

    # Fineweb quality gate (only applied if metadata.score exists)
    ap.add_argument("--fineweb_score_min", type=float, default=3.0)

    # stack-edu configs (REQUIRED if use_stack_edu)
    ap.add_argument(
        "--stack_edu_configs",
        type=str,
        default="Python,Cpp,Go,JavaScript,Rust",
        help=(
            "Comma-separated stack-edu configs. Valid: C,CSharp,Cpp,Go,Java,"
            "JavaScript,Markdown,PHP,Python,Ruby,Rust,SQL,Shell,Swift,TypeScript"
        ),
    )

    args = ap.parse_args()
    hf_token = _hf_token_from_env(args.hf_token)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <|endoftext|> token.")
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    # -------------------------
    # (A) LONGCTX STREAM -> .bin
    # -------------------------
    long_out = BinShardWriter(
        out_dir=os.path.join(args.out_dir, "longctx"),
        prefix="phase3-longctx",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )

    long_sources: list[Iterator[str]] = []
    long_probs: list[float] = []

    if args.use_smollm_corpus:
        fw = load_dataset_streaming(
            "HuggingFaceTB/smollm-corpus",
            config="fineweb-edu-dedup",
            split="train",
            token=hf_token,
        )
        cosmo = load_dataset_streaming(
            "HuggingFaceTB/smollm-corpus",
            config="cosmopedia-v2",
            split="train",
            token=hf_token,
        )

        def fw_iter() -> Iterator[str]:
            for ex in fw:
                t = ex.get("text", "")
                if not isinstance(t, str) or not t.strip():
                    continue
                sc = get_fineweb_score(ex)
                if sc is not None and sc < args.fineweb_score_min:
                    continue
                yield t

        def cosmo_iter() -> Iterator[str]:
            for ex in cosmo:
                t = ex.get("text", "")
                if isinstance(t, str) and t.strip():
                    yield t

        long_sources.extend([fw_iter(), cosmo_iter()])
        long_probs.extend([0.65, 0.25])

    if args.use_stack_edu:
        cfgs = [c.strip() for c in args.stack_edu_configs.split(",") if c.strip()]
        if not cfgs:
            raise ValueError("Empty --stack_edu_configs, but --use_stack_edu was set.")

        stack_iters: list[Iterator[str]] = []
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

            stack_iters.append(_make_iter(ds))

        def stack_mix() -> Iterator[str]:
            return probabilistic_interleave(
                stack_iters,
                [1.0 / len(stack_iters)] * len(stack_iters),
                seed=args.seed + 777,
            )

        long_sources.append(stack_mix())
        long_probs.append(0.10)

    if args.use_finemath:
        fm = load_dataset_streaming(
            "HuggingFaceTB/finemath",
            config="finemath-3plus",
            split="train",
            token=hf_token,
        )

        def fm_iter() -> Iterator[str]:
            for ex in fm:
                t = ex.get("text", "")
                if isinstance(t, str) and t.strip():
                    yield t

        long_sources.append(fm_iter())
        long_probs.append(0.05)

    if not long_sources:
        raise ValueError("No longctx sources selected. Use --use_smollm_corpus at minimum.")

    # Nonblocking interleave (prevents HF stalls from freezing the entire job)
    long_total = sum(long_probs)
    long_probs = [p / max(1e-12, long_total) for p in long_probs]
    long_prefetch = [
        PrefetchIter(name=f"longctx_{i}", it=it, max_buffer=args.prefetch_buffer)
        for i, it in enumerate(long_sources)
    ]
    long_text = interleave_nonblocking(
        sources=long_prefetch,
        probs=long_probs,
        seed=args.seed,
        idle_sleep_sec=0.01,
    )
    
    wrote_long = encode_text_stream_to_bin(
        text_iter=long_text,
        tok=tok,
        eos_id=eos_id,
        out=long_out,
        target_tokens=args.target_tokens_longctx,
        encode_batch_size=args.encode_batch_size,
        flush_interval_sec=args.flush_interval_sec,
        desc="Phase3 LongCtx",
    )
    long_out.finalize()

    # -------------------------
    # (B) SFT STREAM -> .bin
    # -------------------------
    sft_out = BinShardWriter(
        out_dir=os.path.join(args.out_dir, "sft"),
        prefix="phase3-sft",
        shard_size=args.bin_shard_size,
        dtype=dtype,
    )

    sft_sources: list[Iterator[str]] = []
    sft_probs: list[float] = []

    syn = load_dataset("json", data_files=args.synthetic_jsonl, split="train", streaming=True).with_format("python")

    def syn_sft_iter() -> Iterator[str]:
        for ex in syn:
            s = format_synth_chat(ex)
            if s:
                yield s

    # Always include your synthetic data
    sft_sources.append(syn_sft_iter())
    sft_probs.append(0.45)

    if args.use_gsm8k:
        gsm = load_dataset_streaming("gsm8k", config="main", split="train", token=hf_token)

        def gsm_iter() -> Iterator[str]:
            for ex in gsm:
                q = (ex.get("question", "") or "").strip()
                a = (ex.get("answer", "") or "").strip()
                if q and a:
                    yield format_chatml(q, a)

        sft_sources.append(gsm_iter())
        sft_probs.append(0.25)

    if args.use_apps:
        apps = load_dataset_streaming("codeparrot/apps", split="train", token=hf_token)

        def apps_iter() -> Iterator[str]:
            for ex in apps:
                q = (ex.get("question", "") or "").strip()
                sols = ex.get("solutions", None)
                if not q or sols is None:
                    continue
                try:
                    sols_obj = json.loads(sols) if isinstance(sols, str) else sols
                except Exception:
                    continue
                if isinstance(sols_obj, list) and sols_obj:
                    a = str(sols_obj[0]).strip()
                    if a:
                        yield format_chatml(q, a)

        sft_sources.append(apps_iter())
        sft_probs.append(0.20)

    if args.use_finemath:
        fm2 = load_dataset_streaming(
            "HuggingFaceTB/finemath",
            config="finemath-3plus",
            split="train",
            token=hf_token,
        )

        def fm_sft_iter() -> Iterator[str]:
            for ex in fm2:
                t = ex.get("text", "")
                if isinstance(t, str) and t.strip():
                    yield t

        sft_sources.append(fm_sft_iter())
        sft_probs.append(0.10)

    # Normalize SFT probs
    tot = sum(sft_probs)
    sft_probs = [p / tot for p in sft_probs]

    sft_prefetch = [
        PrefetchIter(name=f"sft_{i}", it=it, max_buffer=args.prefetch_buffer)
        for i, it in enumerate(sft_sources)
    ]
    sft_text = interleave_nonblocking(
        sources=sft_prefetch,
        probs=sft_probs,
        seed=args.seed + 1,
        idle_sleep_sec=0.01,
    )
    wrote_sft = encode_text_stream_to_bin(
        text_iter=sft_text,
        tok=tok,
        eos_id=eos_id,
        out=sft_out,
        target_tokens=args.target_tokens_sft,
        encode_batch_size=args.encode_batch_size,
        desc="Phase3 SFT",
    )
    sft_out.finalize()

    # -------------------------
    # (C) DPO PAIRS -> .npz
    # -------------------------
    dpo_out = DpoShardWriter(
        out_dir=os.path.join(args.out_dir, "dpo"),
        prefix="phase3-dpo",
        shard_examples=args.dpo_shard_examples,
        dtype=dtype,
    )

    syn2 = load_dataset("json", data_files=args.synthetic_jsonl, split="train", streaming=True).with_format("python")

    pbar = tqdm(total=args.target_dpo_pairs, unit="pair", desc="Phase3 DPO")
    pairs_written = 0

    for ex in syn2:
        inst = (ex.get("instruction", "") or "").strip()
        inp = (ex.get("input", "") or "").strip()
        chosen_text = (ex.get("output", "") or "").strip()
        if not inst or not chosen_text:
            continue

        user = f"{inst}\n\nInput:\n{inp}" if inp else inst

        h = hashlib.md5((inst + "\n" + chosen_text).encode("utf-8")).hexdigest()
        seed_val = int(h[:8], 16)
        rejected_text = degrade_answer(chosen_text, seed=seed_val)

        prompt = f"<|user|>\n{user}\n<|assistant|>\n"
        prompt_ids = tok.encode(prompt).ids
        chosen_ids = tok.encode(chosen_text).ids
        rejected_ids = tok.encode(rejected_text).ids

        if not prompt_ids or not chosen_ids or not rejected_ids:
            continue

        if chosen_ids[-1] != eos_id:
            chosen_ids.append(eos_id)
        if rejected_ids[-1] != eos_id:
            rejected_ids.append(eos_id)

        dpo_out.push(prompt_ids, chosen_ids, rejected_ids)
        pairs_written += 1
        pbar.update(1)
        if pairs_written >= args.target_dpo_pairs:
            break

    pbar.close()
    dpo_out.finalize()

    print("✅ Phase 3 complete:")
    print(f"  longctx tokens: {wrote_long:,}")
    print(f"  sft tokens:     {wrote_sft:,}")
    print(f"  dpo pairs:      {pairs_written:,}")
    print(f"  out_dir:        {args.out_dir}")


if __name__ == "__main__":
    main()