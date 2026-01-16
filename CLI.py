# CLI.py
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import contextlib
import signal
import threading
import torch
from tokenizers import Tokenizer

from params import Config
from transformer import LLM


_PAUSED = threading.Event()  # when set -> paused
_STOP_GEN = threading.Event()  # stop current generation but keep CLI alive

def _cuda_events_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_initialized()

def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        # MPS autocast is typically float16; bf16 support varies by version
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return contextlib.nullcontext()

@dataclass
class GenParams:
    max_new_tokens: int = 256
    min_new_tokens: int = 0

    temperature: float = 0.5
    top_k: int = 50
    top_p: float = 0.9

    repetition_penalty: float = 1.12
    repetition_window: int = 256

    use_cache: bool = True
    stop_strings: Tuple[str, ...] = ("<|user|>", "<|system|>")


def pretty_piece(text: str) -> str:
    if not text:
        return text
    return (
        text.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", "\n")
        .replace("\r\n", "\n")
    )


def format_chatml(system: str, user: str) -> str:
    system = (system or "").strip()
    user = (user or "").strip()
    out = ""
    if system:
        out += f"<|system|>\n{system}\n"
    out += f"<|user|>\n{user}\n<|assistant|>\n"
    return out


def now_wall() -> float:
    return time.perf_counter()


def cuda_ms(start_evt, end_evt) -> float:
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt)


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

    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "", 1)
        # Drop RoPE caches if present (safe; they are derived, not learned)
        if k.endswith("rope.cos_cached") or k.endswith("rope.sin_cached"):
            continue
        fixed[k] = v

    model.load_state_dict(fixed, strict=False)
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
    Runs the transformer blocks with KV caching properly.
    model.forward() in your code path ignores kv_cache.
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
    logits = logits.float()

    # repetition penalty
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

    if params.temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / params.temperature

    # top-k
    if params.top_k is not None and params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = torch.where(
            logits < thresh,
            torch.full_like(logits, -float("inf")),
            logits,
        )

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


def _truncate_history_to_context(
    tokenizer: Tokenizer,
    system: str,
    turns: List[Tuple[str, str]],
    max_ctx: int,
    reserve_new: int,
) -> List[Tuple[str, str]]:
    """
    Keeps most recent turns so that prompt fits in max_ctx - reserve_new.
    Each turn is (user, assistant_text_so_far).
    """
    budget = max_ctx - max(0, reserve_new)
    budget = max(128, budget)

    def build_text(ts: List[Tuple[str, str]]) -> str:
        out = ""
        if system.strip():
            out += f"<|system|>\n{system.strip()}\n"
        for u, a in ts:
            out += f"<|user|>\n{u.strip()}\n<|assistant|>\n{a.strip()}\n"
        return out

    # Keep trimming oldest until it fits
    cur = turns[:]
    while cur:
        txt = build_text(cur)
        n = len(tokenizer.encode(txt).ids)
        if n <= budget:
            return cur
        cur = cur[1:]
    return turns[-1:] if turns else []


def _contains_stop(text_tail: str, stop_strings: Tuple[str, ...]) -> bool:
    for s in stop_strings:
        if s and s in text_tail:
            return True
    return False


@torch.no_grad()
def generate_chat(
    model: LLM,
    tokenizer: Tokenizer,
    system: str,
    turns: List[Tuple[str, str]],
    user_msg: str,
    params: GenParams,
    device: torch.device,
    bench: bool,
    raw_prompt: bool,
) -> Tuple[str, dict]:
    """
    Returns (assistant_text, stats)
    """
    _STOP_GEN.clear()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = -1

    max_ctx = getattr(model, "max_seq_len", getattr(Config, "max_seq_len", 8192))

    if raw_prompt:
        prompt_text = user_msg
    else:
        # Truncate history for context
        trimmed = _truncate_history_to_context(
            tokenizer=tokenizer,
            system=system,
            turns=turns,
            max_ctx=max_ctx,
            reserve_new=params.max_new_tokens + 32,
        )
        # Append new user turn with empty assistant (to be generated)
        all_turns = trimmed + [(user_msg, "")]
        # Build ChatML prompt where last assistant is empty
        prompt_text = ""
        if system.strip():
            prompt_text += f"<|system|>\n{system.strip()}\n"
        for i, (u, a) in enumerate(all_turns):
            prompt_text += f"<|user|>\n{u.strip()}\n<|assistant|>\n"
            if i < len(all_turns) - 1:
                prompt_text += f"{a.strip()}\n"

    prompt_ids = tokenizer.encode(prompt_text).ids
    if not prompt_ids:
        return "", {"num_prompt_tokens": 0, "num_new_tokens": 0}

    if len(prompt_ids) > max_ctx:
        prompt_ids = prompt_ids[-max_ctx:]

    torch.set_float32_matmul_precision("high")

    wall_t0 = now_wall()
    gpu_total_ms = 0.0 if _cuda_events_available() else None

    # Prefill
    idx = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    if _cuda_events_available():
        pre_s = torch.cuda.Event(enable_timing=True)
        pre_e = torch.cuda.Event(enable_timing=True)
        pre_s.record()

    with torch.inference_mode(), _autocast_ctx(device):
        logits, kv_cache = forward_with_kv_cache(
            model, idx=idx, kv_cache=None, use_cache=params.use_cache
        )

    if _cuda_events_available():
        pre_ms = cuda_ms(pre_s, pre_e)
        gpu_total_ms += pre_ms

    wall_after_prefill = now_wall()
    first_token_wall: Optional[float] = None

    generated: List[int] = []
    out_text = ""
    recent_tail = ""

    one = torch.empty((1, 1), device=device, dtype=torch.long)
    next_logits = logits[:, -1, :]

    for step in range(int(params.max_new_tokens)):
        
        # allow pause/resume without exiting
        while _PAUSED.is_set():
            time.sleep(0.05)

        if _STOP_GEN.is_set():
            break

        next_id = sample_next_token(next_logits, generated, params)

        if first_token_wall is None:
            first_token_wall = now_wall()

        generated.append(next_id)

        if next_id == eos_id and step >= params.min_new_tokens:
            break

        piece = tokenizer.decode([next_id])
        piece = pretty_piece(piece)
        out_text += piece
        # Keep a small tail for stop detection
        recent_tail = (recent_tail + piece)[-512:]

        if (step + 1) >= params.min_new_tokens and params.stop_strings:
            if _contains_stop(recent_tail, params.stop_strings):
                # Stop before printing the stop sequence too deeply.
                break

        if not bench and piece:
            sys.stdout.write(piece)
            sys.stdout.flush()

        one[0, 0] = next_id

        if _cuda_events_available():
            dec_s = torch.cuda.Event(enable_timing=True)
            dec_e = torch.cuda.Event(enable_timing=True)
            dec_s.record()

        with torch.inference_mode(), _autocast_ctx(device):
            logits, kv_cache = forward_with_kv_cache(
                model, idx=one, kv_cache=kv_cache, use_cache=params.use_cache
            )

        if _cuda_events_available():
            dec_ms = cuda_ms(dec_s, dec_e)
            gpu_total_ms += dec_ms

        next_logits = logits[:, -1, :]

    wall_t1 = now_wall()

    num_new = len(generated)
    wall_total = wall_t1 - wall_t0
    wall_prefill = wall_after_prefill - wall_t0
    wall_decode = wall_t1 - wall_after_prefill

    ttft_ms = 0.0
    if first_token_wall is not None:
        ttft_ms = (first_token_wall - wall_t0) * 1000.0

    out_stats = {
        "num_prompt_tokens": len(prompt_ids),
        "num_new_tokens": num_new,
        "ttft_ms": ttft_ms,
        "prefill_tps_wall": len(prompt_ids) / max(wall_prefill, 1e-9),
        "decode_tps_wall": num_new / max(wall_decode, 1e-9),
        "wall_total_s": wall_total,
    }
    if gpu_total_ms is not None:
        out_stats["gpu_total_s"] = gpu_total_ms / 1000.0

    return out_text, out_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/best_step_240.pt")
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer.json")
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    ap.add_argument("--device", type=str, default=default_device)

    ap.add_argument("--bench", action="store_true", help="Don’t stream; just speed.")
    ap.add_argument("--raw", action="store_true", help="Don’t wrap in ChatML.")
    ap.add_argument(
        "--system",
        type=str,
        default=(
            "You are a helpful, concise assistant. Follow the user's instructions "
            "exactly. Ask clarifying questions when needed."
        ),
    )
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--min_new", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--topp", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.12)
    ap.add_argument("--rep_window", type=int, default=256)

    ap.add_argument("--stop", type=str, default="<|user|>,<|system|>")
    ap.add_argument("--no_stop", action="store_true")

    ap.add_argument("--save_chat", type=str, default=None)
    ap.add_argument("--load_chat", type=str, default=None)

    args = ap.parse_args()

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    def _handle_sigint(sig, frame):
        # Stop current generation, but don't exit CLI
        _STOP_GEN.set()

    def _handle_sigtstp(sig, frame):
        # Toggle pause/resume
        if _PAUSED.is_set():
            _PAUSED.clear()
            sys.stdout.write("\n[resumed]\n")
            sys.stdout.flush()
        else:
            _PAUSED.set()
            sys.stdout.write("\n[paused: Ctrl+Z to resume]\n")
            sys.stdout.flush()

    signal.signal(signal.SIGINT, _handle_sigint)
    # Ctrl+Z sends SIGTSTP on macOS/Linux terminals
    signal.signal(signal.SIGTSTP, _handle_sigtstp)

    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, device=device)

    params = GenParams(
        max_new_tokens=int(args.max_new),
        min_new_tokens=int(args.min_new),
        temperature=float(args.temp),
        top_k=int(args.topk),
        top_p=float(args.topp),
        repetition_penalty=float(args.rep),
        repetition_window=int(args.rep_window),
        use_cache=True,
        stop_strings=tuple(
            s.strip() for s in (args.stop.split(",") if args.stop else []) if s.strip()
        ),
    )
    if args.no_stop:
        params.stop_strings = tuple()

    # Conversation state: list of (user, assistant)
    turns: List[Tuple[str, str]] = []
    system = args.system
    raw_prompt = bool(args.raw)
    bench = bool(args.bench)

    if args.load_chat:
        p = Path(args.load_chat)
        obj = json.loads(p.read_text())
        system = obj.get("system", system)
        turns = [(t["user"], t["assistant"]) for t in obj.get("turns", [])]

    print("\n=== CHAT CLI (ChatML-wrapped) ===")
    print(
        "Commands:\n"
        "  /system <text>    set system prompt\n"
        "  /reset            clear history\n"
        "  /raw 0|1          toggle raw prompt (no ChatML)\n"
        "  /bench 0|1        toggle bench mode\n"
        "  /max N            max_new_tokens\n"
        "  /min N            min_new_tokens\n"
        "  /temp x           temperature\n"
        "  /topk k           top_k\n"
        "  /topp p           top_p\n"
        "  /rep r            repetition_penalty\n"
        "  /stop a,b,c       set stop strings\n"
        "  /nostop           disable stop strings\n"
        "  /show             show current settings\n"
        "  /save path.json   save chat\n"
        "  /load path.json   load chat\n"
        "  exit\n"
    )

    while True:
        try:
            user_msg = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break

        if user_msg.startswith("/"):
            parts = user_msg.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) == 2 else ""

            try:
                if cmd == "/system":
                    system = arg
                elif cmd == "/reset":
                    turns = []
                elif cmd == "/raw":
                    raw_prompt = bool(int(arg.strip() or "0"))
                elif cmd == "/bench":
                    bench = bool(int(arg.strip() or "0"))
                elif cmd == "/max":
                    params.max_new_tokens = int(arg)
                elif cmd == "/min":
                    params.min_new_tokens = int(arg)
                elif cmd == "/temp":
                    params.temperature = float(arg)
                elif cmd == "/topk":
                    params.top_k = int(arg)
                elif cmd == "/topp":
                    params.top_p = float(arg)
                elif cmd == "/rep":
                    params.repetition_penalty = float(arg)
                elif cmd == "/stop":
                    params.stop_strings = tuple(
                        s.strip() for s in arg.split(",") if s.strip()
                    )
                elif cmd == "/nostop":
                    params.stop_strings = tuple()
                elif cmd == "/show":
                    print(
                        f"raw={int(raw_prompt)} bench={int(bench)} "
                        f"max_new={params.max_new_tokens} min_new={params.min_new_tokens} "
                        f"temp={params.temperature} topk={params.top_k} topp={params.top_p} "
                        f"rep={params.repetition_penalty} rep_window={params.repetition_window} "
                        f"stop={list(params.stop_strings)}"
                    )
                    continue
                elif cmd == "/save":
                    path = arg.strip()
                    if not path:
                        print("Usage: /save path.json")
                        continue
                    obj = {
                        "system": system,
                        "turns": [{"user": u, "assistant": a} for u, a in turns],
                    }
                    Path(path).write_text(json.dumps(obj, indent=2))
                    print(f"Saved: {path}")
                    continue
                elif cmd == "/load":
                    path = arg.strip()
                    if not path:
                        print("Usage: /load path.json")
                        continue
                    obj = json.loads(Path(path).read_text())
                    system = obj.get("system", system)
                    turns = [(t["user"], t["assistant"]) for t in obj.get("turns", [])]
                    print(f"Loaded: {path}")
                    continue
                else:
                    print("Unknown command.")
                    continue
            except Exception as e:
                print(f"Bad command: {e}")
                continue

            print("OK.")
            continue

        print("Completion: ", end="" if not bench else "\n", flush=True)

        assistant_text, stats = generate_chat(
            model=model,
            tokenizer=tokenizer,
            system=system,
            turns=turns,
            user_msg=user_msg,
            params=params,
            device=device,
            bench=bench,
            raw_prompt=raw_prompt,
        )
        if not bench:
            print()

        # Append to history (only in ChatML mode)
        if not raw_prompt:
            turns.append((user_msg, assistant_text))

        print(
            f"[prompt={stats['num_prompt_tokens']} new={stats['num_new_tokens']}] "
            f"TTFT={stats['ttft_ms']:.1f}ms | "
            f"prefill={stats['prefill_tps_wall']:.1f} tok/s | "
            f"decode={stats['decode_tps_wall']:.1f} tok/s | "
            f"wall={stats['wall_total_s']:.2f}s"
        )


if __name__ == "__main__":
    main()