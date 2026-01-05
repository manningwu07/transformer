# CLI.py
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from params import Config
from transformer import LLM


def _cuda_events_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_initialized()


@dataclass
class GenParams:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.12
    repetition_window: int = 256
    use_cache: bool = True


def pretty_piece(text: str) -> str:
    """
    Your tokenizer appears to use GPT2/RoBERTa-style markers:
      - 'Ġ' for "leading space"
      - 'Ċ' for newline
    Some SentencePiece tokenizers use '▁' for space.
    """
    if not text:
        return text
    return (
        text.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", "\n")
        .replace("\r\n", "\n")
    )


def load_model(ckpt_path: str, device: torch.device) -> LLM:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Ensure inference doesn’t accidentally trigger train-time toggles
    Config.use_float8 = False
    Config.compile_mode = "none"
    Config.gradient_checkpointing = False

    model = LLM(Config).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Strip torch.compile prefix if present
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
    """
    IMPORTANT:
    Your current LLM.forward() ignores kv_cache/use_cache and always calls
    forward_no_cache() on each layer, so generation loses context and “cache”
    doesn’t work. This function runs the blocks with KV caching properly.
    """
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


def sample_next_token(
    logits: torch.Tensor,
    generated: List[int],
    params: GenParams,
) -> int:
    # logits: [1, vocab]
    logits = logits.float()

    # repetition penalty on a window (cheap + helps loops)
    if params.repetition_penalty != 1.0 and generated:
        window = generated[-params.repetition_window :]
        uniq = set(window)
        if uniq:
            ids = torch.tensor(list(uniq), device=logits.device, dtype=torch.long)
            vals = logits[0, ids]
            pos = vals > 0
            vals[pos] = vals[pos] / params.repetition_penalty
            vals[~pos] = vals[~pos] * params.repetition_penalty
            logits[0, ids] = vals

    # greedy if temperature <= 0
    if params.temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / params.temperature

    # top-k
    if params.top_k is not None and params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)

    # top-p
    if params.top_p is not None and params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)

        cut = cdf > params.top_p
        cut[:, 1:] = cut[:, :-1].clone()
        cut[:, 0] = False

        to_remove = cut.scatter(1, sorted_idx, cut)
        logits = logits.masked_fill(to_remove, -float("inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def now_wall() -> float:
    return time.perf_counter()


def cuda_ms(start_evt, end_evt) -> float:
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt)


@torch.no_grad()
def generate_stream(
    model: LLM,
    tokenizer: Tokenizer,
    prompt: str,
    params: GenParams,
    device: torch.device,
    bench: bool = False,
) -> dict:
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        # If missing, disable EOS stopping
        eos_id = -1

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        return {
            "num_new_tokens": 0,
            "ttft_ms": 0.0,
            "prefill_tps": 0.0,
            "decode_tps": 0.0,
            "wall_total_s": 0.0,
            "gpu_total_s": None,
        }

    max_ctx = getattr(model, "max_seq_len", getattr(Config, "max_seq_len", 8192))
    if len(prompt_ids) > max_ctx:
        prompt_ids = prompt_ids[-max_ctx:]

    # small speed knobs
    torch.set_float32_matmul_precision("high")

    # timing setup
    wall_t0 = now_wall()
    gpu_total_ms = 0.0 if _cuda_events_available() else None

    # ---- Prefill (build cache on full prompt)
    idx = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    if _cuda_events_available():
        pre_s = torch.cuda.Event(enable_timing=True)
        pre_e = torch.cuda.Event(enable_timing=True)
        pre_s.record()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, kv_cache = forward_with_kv_cache(
            model, idx=idx, kv_cache=None, use_cache=params.use_cache
        )

    if _cuda_events_available():
        pre_ms = cuda_ms(pre_s, pre_e)
        gpu_total_ms += pre_ms

    # ---- Decode tokens
    generated: List[int] = []
    printed = 0

    # reuse a 1x1 tensor to avoid per-step allocations
    one = torch.empty((1, 1), device=device, dtype=torch.long)

    wall_after_prefill = now_wall()
    first_token_wall: Optional[float] = None

    # initial logits from prefill last position
    next_logits = logits[:, -1, :]

    for _ in range(int(params.max_new_tokens)):
        next_id = sample_next_token(next_logits, generated, params)

        if first_token_wall is None:
            first_token_wall = now_wall()

        if next_id == eos_id:
            break

        generated.append(next_id)

        # stream-print
        if not bench:
            # decode one token and pretty it (fast + readable for Ġ/Ċ tokenizers)
            piece = tokenizer.decode([next_id])
            piece = pretty_piece(piece)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
                printed += len(piece)

        # forward one token with cache
        one[0, 0] = next_id

        if _cuda_events_available():
            dec_s = torch.cuda.Event(enable_timing=True)
            dec_e = torch.cuda.Event(enable_timing=True)
            dec_s.record()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, kv_cache = forward_with_kv_cache(
                model, idx=one, kv_cache=kv_cache, use_cache=params.use_cache
            )

        if _cuda_events_available():
            dec_ms = cuda_ms(dec_s, dec_e)
            gpu_total_ms += dec_ms

        next_logits = logits[:, -1, :]

        # simple repetition-loop detector
        if len(generated) >= 32:
            a = generated[-32:-16]
            b = generated[-16:]
            if a == b:
                if not bench:
                    sys.stdout.write("\n[stopped: repetition detected]\n")
                    sys.stdout.flush()
                break

    wall_t1 = now_wall()

    # metrics
    num_new = len(generated)
    wall_total = wall_t1 - wall_t0
    wall_prefill = wall_after_prefill - wall_t0
    wall_decode = wall_t1 - wall_after_prefill

    ttft_ms = 0.0
    if first_token_wall is not None:
        ttft_ms = (first_token_wall - wall_t0) * 1000.0

    prefill_tps = len(prompt_ids) / max(wall_prefill, 1e-9)
    decode_tps = num_new / max(wall_decode, 1e-9)

    out = {
        "num_prompt_tokens": len(prompt_ids),
        "num_new_tokens": num_new,
        "ttft_ms": ttft_ms,
        "prefill_tps_wall": prefill_tps,
        "decode_tps_wall": decode_tps,
        "wall_total_s": wall_total,
    }

    if gpu_total_ms is not None:
        out["gpu_total_s"] = gpu_total_ms / 1000.0
        out["decode_tps_gpu"] = num_new / max((gpu_total_ms / 1000.0) - (wall_prefill), 1e-9)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/phase-one-model.pt")
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--bench", action="store_true", help="Don’t stream text; just measure speed")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--topp", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.12)

    args = ap.parse_args()

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, device=device)

    print("\n=== 1B MLA MODEL - CLI INFERENCE ===")
    print("Commands: /temp x | /topk k | /topp p | /rep r | /max n | /bench 0/1 | exit\n")

    params = GenParams(
        max_new_tokens=int(args.max_new),
        temperature=float(args.temp),
        top_k=int(args.topk),
        top_p=float(args.topp),
        repetition_penalty=float(args.rep),
        use_cache=True,
    )
    bench = bool(args.bench)

    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return

        # runtime commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()

            try:
                if cmd == "/temp" and len(parts) == 2:
                    params.temperature = float(parts[1])
                elif cmd == "/topk" and len(parts) == 2:
                    params.top_k = int(parts[1])
                elif cmd == "/topp" and len(parts) == 2:
                    params.top_p = float(parts[1])
                elif cmd == "/rep" and len(parts) == 2:
                    params.repetition_penalty = float(parts[1])
                elif cmd == "/max" and len(parts) == 2:
                    params.max_new_tokens = int(parts[1])
                elif cmd == "/bench" and len(parts) == 2:
                    bench = bool(int(parts[1]))
                else:
                    print("Unknown command.")
                    continue
            except Exception as e:
                print(f"Bad command: {e}")
                continue

            print(
                f"Params: temp={params.temperature} topk={params.top_k} "
                f"topp={params.top_p} rep={params.repetition_penalty} "
                f"max_new={params.max_new_tokens} bench={int(bench)}"
            )
            continue

        print("Completion: ", end="" if not bench else "\n", flush=True)
        stats = generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            params=params,
            device=device,
            bench=bench,
        )
        if not bench:
            print()

        print(
            f"[tokens prompt={stats['num_prompt_tokens']} new={stats['num_new_tokens']}] "
            f"TTFT={stats['ttft_ms']:.1f}ms | "
            f"prefill={stats['prefill_tps_wall']:.1f} tok/s | "
            f"decode={stats['decode_tps_wall']:.1f} tok/s | "
            f"wall={stats['wall_total_s']:.2f}s"
        )


if __name__ == "__main__":
    main()