# CLI.py

# Make sure to increase mem block alloc to decrease num of allocs during generation:
# export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
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
    if not text:
        return text
    return (
        text.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", "\n")
        .replace("\r\n", "\n")
    )


def load_model(ckpt_path: str, device: torch.device, compile_model: bool = True) -> LLM:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    Config.use_float8 = False
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

    if compile_model and torch.cuda.is_available():
        print("Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    return model


class StaticKVCache:
    """Pre-allocated KV cache to avoid per-token allocations."""

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pos = 0

        # Shape: [layers, 2, 1, max_seq, heads, head_dim]
        self.cache = torch.zeros(
            (num_layers, 2, 1, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def reset(self):
        self.pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k, v: [batch, seq_len, heads, head_dim]
        Returns full k, v up to current position.
        """
        seq_len = k.size(1)
        end_pos = self.pos + seq_len

        self.cache[layer_idx, 0, :, self.pos:end_pos] = k
        self.cache[layer_idx, 1, :, self.pos:end_pos] = v

        return (
            self.cache[layer_idx, 0, :, :end_pos],
            self.cache[layer_idx, 1, :, :end_pos],
        )

    def advance(self, seq_len: int):
        self.pos += seq_len


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


class TokenSampler:
    """Vectorized sampler with pre-allocated tensors."""

    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        # Pre-allocate penalty buffer
        self.penalty_buf = torch.ones(vocab_size, device=device, dtype=torch.float32)
        # Pre-allocate generated tokens buffer
        self.generated = torch.zeros(8192, device=device, dtype=torch.long)
        self.gen_len = 0

    def reset(self):
        self.gen_len = 0

    def add_token(self, token_id: int):
        if self.gen_len < self.generated.size(0):
            self.generated[self.gen_len] = token_id
            self.gen_len += 1

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        rep_penalty: float,
        rep_window: int,
    ) -> int:
        logits = logits.float()

        # Vectorized repetition penalty
        if rep_penalty != 1.0 and self.gen_len > 0:
            window_start = max(0, self.gen_len - rep_window)
            window = self.generated[window_start : self.gen_len]
            unique_ids = torch.unique(window)

            self.penalty_buf.fill_(1.0)
            self.penalty_buf[unique_ids] = rep_penalty

            positive = logits[0] > 0
            logits[0] = torch.where(
                positive,
                logits[0] / self.penalty_buf,
                logits[0] * self.penalty_buf,
            )

        # Greedy
        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / temperature

        # Top-k
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k, dim=-1)
            thresh = topk_vals[:, -1:]
            logits = torch.where(
                logits < thresh,
                torch.tensor(-float("inf"), device=logits.device),
                logits,
            )

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(sorted_probs, dim=-1)

            cut = cdf > top_p
            cut[:, 1:] = cut[:, :-1].clone()
            cut[:, 0] = False

            to_remove = cut.scatter(1, sorted_idx, cut)
            logits = logits.masked_fill(to_remove, -float("inf"))

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class OutputBuffer:
    """Buffers output to reduce SSH flush overhead."""

    def __init__(self, flush_interval: int = 4, flush_time_ms: float = 50.0):
        self.buffer: List[str] = []
        self.flush_interval = flush_interval
        self.flush_time_ms = flush_time_ms
        self.last_flush = time.perf_counter()
        self.token_count = 0

    def write(self, text: str):
        self.buffer.append(text)
        self.token_count += 1

        now = time.perf_counter()
        elapsed_ms = (now - self.last_flush) * 1000

        if self.token_count >= self.flush_interval or elapsed_ms >= self.flush_time_ms:
            self.flush()

    def flush(self):
        if self.buffer:
            sys.stdout.write("".join(self.buffer))
            sys.stdout.flush()
            self.buffer.clear()
            self.token_count = 0
            self.last_flush = time.perf_counter()


@torch.no_grad()
def generate_stream(
    model: LLM,
    tokenizer: Tokenizer,
    prompt: str,
    params: GenParams,
    device: torch.device,
    sampler: TokenSampler,
    bench: bool = False,
) -> dict:
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = -1

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        return {
            "num_new_tokens": 0,
            "ttft_ms": 0.0,
            "prefill_tps": 0.0,
            "decode_tps": 0.0,
            "wall_total_s": 0.0,
        }

    max_ctx = getattr(model, "max_seq_len", getattr(Config, "max_seq_len", 8192))
    if len(prompt_ids) > max_ctx:
        prompt_ids = prompt_ids[-max_ctx:]

    torch.set_float32_matmul_precision("high")

    # Reset sampler state
    sampler.reset()

    wall_t0 = time.perf_counter()

    # Prefill
    idx = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, kv_cache = forward_with_kv_cache(
            model, idx=idx, kv_cache=None, use_cache=params.use_cache
        )

    wall_after_prefill = time.perf_counter()

    # Decode
    output_buf = OutputBuffer(flush_interval=4, flush_time_ms=50.0)
    one = torch.empty((1, 1), device=device, dtype=torch.long)
    next_logits = logits[:, -1, :]

    first_token_wall: Optional[float] = None
    num_generated = 0

    # Buffer for multi-token decoding
    decode_buffer: List[int] = []

    for _ in range(int(params.max_new_tokens)):
        next_id = sampler.sample(
            next_logits,
            params.temperature,
            params.top_k,
            params.top_p,
            params.repetition_penalty,
            params.repetition_window,
        )

        if first_token_wall is None:
            first_token_wall = time.perf_counter()

        if next_id == eos_id:
            break

        sampler.add_token(next_id)
        num_generated += 1

        # Buffered decoding for better multi-byte handling
        if not bench:
            decode_buffer.append(next_id)
            if len(decode_buffer) >= 3 or next_id in (198, 628):  # newline tokens
                piece = tokenizer.decode(decode_buffer)
                piece = pretty_piece(piece)
                if piece:
                    output_buf.write(piece)
                decode_buffer.clear()

        one[0, 0] = next_id

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, kv_cache = forward_with_kv_cache(
                model, idx=one, kv_cache=kv_cache, use_cache=params.use_cache
            )

        next_logits = logits[:, -1, :]

        # Repetition loop detection (on GPU tensor now)
        if num_generated >= 32:
            recent = sampler.generated[num_generated - 32 : num_generated]
            if torch.equal(recent[:16], recent[16:]):
                if not bench:
                    output_buf.write("\n[stopped: repetition detected]\n")
                break

    # Flush remaining decode buffer
    if not bench and decode_buffer:
        piece = tokenizer.decode(decode_buffer)
        piece = pretty_piece(piece)
        if piece:
            output_buf.write(piece)
    
    if not bench:
        output_buf.flush()

    wall_t1 = time.perf_counter()

    # Only sync for final timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_total = wall_t1 - wall_t0
    wall_prefill = wall_after_prefill - wall_t0
    wall_decode = wall_t1 - wall_after_prefill

    ttft_ms = 0.0
    if first_token_wall is not None:
        ttft_ms = (first_token_wall - wall_t0) * 1000.0

    prefill_tps = len(prompt_ids) / max(wall_prefill, 1e-9)
    decode_tps = num_generated / max(wall_decode, 1e-9)

    return {
        "num_prompt_tokens": len(prompt_ids),
        "num_new_tokens": num_generated,
        "ttft_ms": ttft_ms,
        "prefill_tps_wall": prefill_tps,
        "decode_tps_wall": decode_tps,
        "wall_total_s": wall_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/phase-one-model.pt")
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    ap.add_argument("--bench", action="store_true", help="Don't stream text; just measure speed")
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
    model = load_model(args.ckpt, device=device, compile_model=not args.no_compile)

    # Get vocab size for sampler
    vocab_size = tokenizer.get_vocab_size()
    sampler = TokenSampler(vocab_size, device)

    print("\n=== 1B MLA MODEL - CLI INFERENCE (OPTIMIZED) ===")
    print("Commands: /temp x | /topk k | /topp p | /rep r | /max n | /bench 0/1 | exit\n")

    # Warmup pass (important for torch.compile)
    if not args.no_compile and torch.cuda.is_available():
        print("Warming up compiled model...")
        _ = generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt="Hello",
            params=GenParams(max_new_tokens=8),
            device=device,
            sampler=sampler,
            bench=True,
        )
        print("Warmup complete.\n")

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
            sampler=sampler,
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