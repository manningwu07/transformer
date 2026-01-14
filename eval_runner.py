#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from params import Config
from transformer import LLM


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def load_model(ckpt_path: str, device: torch.device) -> LLM:
    Config.use_float8 = False
    Config.compile_mode = "none"
    Config.gradient_checkpointing = False

    model = LLM(Config).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "", 1)
        fixed[k] = v

    model.load_state_dict(fixed, strict=True)
    model.eval()
    return model


@torch.no_grad()
def forward_with_kv_cache(
    model: LLM,
    idx: torch.Tensor,
    kv_cache: Optional[List[Optional[dict]]],
    use_cache: bool,
) -> Tuple[torch.Tensor, Optional[List[Optional[dict]]]]:
    x = model.tok_embeddings(idx)

    new_caches: Optional[List[Optional[dict]]] = [] if use_cache else None
    if kv_cache is None and use_cache:
        kv_cache = [None] * len(model.layers)

    for i, layer in enumerate(model.layers):
        if use_cache:
            x, layer_cache = layer(x, kv_cache=kv_cache[i], use_cache=True)
            new_caches.append(layer_cache)
        else:
            x = layer.forward_no_cache(x)

    x = model.norm(x)
    logits = model.output(x)
    return logits, new_caches


@dataclass
class GenParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # greedy
    use_cache: bool = True


def greedy_next_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def generate_text(
    model: LLM,
    tok: Tokenizer,
    prompt: str,
    device: torch.device,
    params: GenParams,
) -> str:
    prompt_ids = tok.encode(prompt).ids
    if not prompt_ids:
        return ""

    max_ctx = getattr(model, "max_seq_len", getattr(Config, "max_seq_len", 2048))
    if len(prompt_ids) > max_ctx:
        prompt_ids = prompt_ids[-max_ctx:]

    idx = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    with torch.inference_mode(), autocast_ctx(device):
        logits, kv_cache = forward_with_kv_cache(
            model, idx=idx, kv_cache=None, use_cache=params.use_cache
        )

    out_ids: List[int] = []
    one = torch.empty((1, 1), device=device, dtype=torch.long)
    next_logits = logits[:, -1, :]

    for _ in range(int(params.max_new_tokens)):
        nid = greedy_next_token(next_logits)
        out_ids.append(nid)
        one[0, 0] = nid
        with torch.inference_mode(), autocast_ctx(device):
            logits, kv_cache = forward_with_kv_cache(
                model, idx=one, kv_cache=kv_cache, use_cache=params.use_cache
            )
        next_logits = logits[:, -1, :]

    return tok.decode(out_ids)


def read_prompts_file(path: str) -> List[str]:
    prompts: List[str] = []
    buf: List[str] = []
    with open(path, "r") as f:
        for line in f:
            if line.strip() == "---":
                p = "".join(buf).strip()
                if p:
                    prompts.append(p)
                buf = []
            else:
                buf.append(line)
    p = "".join(buf).strip()
    if p:
        prompts.append(p)
    return prompts


def extract_last_int(text: str) -> Optional[int]:
    nums = re.findall(r"-?\d+", text)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None


def eval_arithmetic(
    model: LLM,
    tok: Tokenizer,
    device: torch.device,
    seed: int,
    n: int,
) -> Dict:
    rng = random.Random(seed)
    correct = 0
    examples = []

    for i in range(n):
        a = rng.randint(-10_000, 10_000)
        b = rng.randint(-10_000, 10_000)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b

        prompt = f"Q: Compute {a} {op} {b}.\nA:"
        out = generate_text(model, tok, prompt, device, GenParams(max_new_tokens=32))
        pred = extract_last_int(out)
        ok = pred == ans
        correct += 1 if ok else 0
        if i < 10:
            examples.append(
                {
                    "prompt": prompt,
                    "expected": ans,
                    "completion": out,
                    "pred": pred,
                    "ok": ok,
                }
            )

    return {
        "task": "arithmetic",
        "n": n,
        "acc": correct / max(1, n),
        "examples": examples,
    }


def make_needle_prompt(
    rng: random.Random,
    needle: str,
    target_total_tokens: int,
    tok: Tokenizer,
) -> str:
    # Build filler from a small vocabulary so tokenization is stable-ish.
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "lambda"]
    parts = []
    parts.append("You will be given a long text. Memorize the secret token.\n\n")
    parts.append(f"SECRET_TOKEN = {needle}\n\n")
    # Pad until token count exceeds target_total_tokens - reserve for question.
    while len(tok.encode("".join(parts)).ids) < max(0, target_total_tokens - 64):
        parts.append(" ".join(rng.choice(words) for _ in range(32)) + "\n")
    parts.append("\nQuestion: What is the SECRET_TOKEN?\nAnswer:")
    return "".join(parts)


def eval_needle(
    model: LLM,
    tok: Tokenizer,
    device: torch.device,
    seed: int,
    n: int,
    distances: List[int],
) -> Dict:
    rng = random.Random(seed)
    results = []
    for dist in distances:
        correct = 0
        for i in range(n):
            needle = f"ZXQ-{rng.randint(1000, 9999)}"
            prompt = make_needle_prompt(rng, needle=needle, target_total_tokens=dist, tok=tok)
            out = generate_text(model, tok, prompt, device, GenParams(max_new_tokens=24))
            ok = needle in out
            correct += 1 if ok else 0
        results.append({"target_tokens": dist, "n": n, "acc_contains": correct / max(1, n)})
    return {"task": "needle", "results": results}


def eval_prompt_suite(
    model: LLM,
    tok: Tokenizer,
    device: torch.device,
    prompts_file: str,
    max_new: int,
) -> Dict:
    prompts = read_prompts_file(prompts_file)
    outs = []
    for i, p in enumerate(prompts):
        out = generate_text(model, tok, p, device, GenParams(max_new_tokens=max_new))
        outs.append({"i": i, "prompt": p, "completion": out})
    return {"task": "prompt_suite", "n": len(prompts), "outputs": outs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer_32k.json")
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--out_json", type=str, default="eval_runs/eval.json")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--do_arithmetic", action="store_true")
    ap.add_argument("--arith_n", type=int, default=200)

    ap.add_argument("--do_needle", action="store_true")
    ap.add_argument("--needle_n", type=int, default=50)
    ap.add_argument("--needle_distances", type=str, default="256,512,1024,1536,2048")

    ap.add_argument("--prompts_file", type=str, default=None)
    ap.add_argument("--suite_max_new", type=int, default=128)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    device = pick_device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)

    t0 = time.time()
    model = load_model(args.ckpt, device=device)
    t1 = time.time()

    report = {
        "ckpt": args.ckpt,
        "tokenizer": args.tokenizer,
        "device": str(device),
        "load_s": t1 - t0,
        "time": time.time(),
        "results": [],
    }

    if args.do_arithmetic:
        report["results"].append(
            eval_arithmetic(model, tok, device, seed=args.seed, n=args.arith_n)
        )

    if args.do_needle:
        dists = [int(x) for x in args.needle_distances.split(",") if x.strip()]
        report["results"].append(
            eval_needle(model, tok, device, seed=args.seed + 1, n=args.needle_n, distances=dists)
        )

    if args.prompts_file:
        report["results"].append(
            eval_prompt_suite(
                model,
                tok,
                device,
                prompts_file=args.prompts_file,
                max_new=args.suite_max_new,
            )
        )

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote: {args.out_json}")
    for r in report["results"]:
        if r["task"] == "arithmetic":
            print(f"arithmetic acc: {r['acc']:.3f} (n={r['n']})")
        if r["task"] == "needle":
            for row in r["results"]:
                print(
                    f"needle @ {row['target_tokens']} tok: acc_contains={row['acc_contains']:.3f} (n={row['n']})"
                )


if __name__ == "__main__":
    main()