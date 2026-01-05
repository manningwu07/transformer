# CLI.py
import os
import time
import torch
from tokenizers import Tokenizer
from transformer import LLM
from params import Config

# --- CONFIGURATION ---
CHECKPOINT_PATH = "models/phase-one-model.pt"
TOKENIZER_PATH = "data/json/tokenizer_32k.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path):
    print(f"📦 Loading model from {ckpt_path}...")
    
    Config.use_float8 = False
    Config.compile_mode = "none"
    model = LLM(Config).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model", ckpt)  # Handle both formats
    
    # Strip compiled prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_with_metrics(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    use_cache: bool = True,
):
    """
    Generate text with proper decoding, repetition penalty, and TPS logging.
    """
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], device=DEVICE, dtype=torch.long)
    
    generated_ids = []
    kv_cache = None
    
    # Track timing
    start_time = time.perf_counter()
    first_token_time = None
    
    for i in range(max_new_tokens):
        # Prepare input
        if use_cache and kv_cache is not None:
            idx_input = torch.tensor([[generated_ids[-1]]], device=DEVICE, dtype=torch.long)
        else:
            all_ids = input_ids + generated_ids
            idx_input = torch.tensor([all_ids[-Config.max_seq_len:]], device=DEVICE, dtype=torch.long)
        
        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _, kv_cache = model(idx_input, use_cache=use_cache, kv_cache=kv_cache)
        
        logits = logits[:, -1, :].float()  # [1, vocab_size]
        
        # === Apply Repetition Penalty ===
        if repetition_penalty != 1.0 and generated_ids:
            # Penalize tokens that have already appeared
            unique_tokens = set(generated_ids)
            for token_id in unique_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        
        # === Temperature ===
        if temperature > 0:
            logits = logits / temperature
        
        # === Top-K Filtering ===
        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
            logits[indices_to_remove] = float('-inf')
        
        # === Top-P (Nucleus) Filtering ===
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # === Sample ===
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Record first token time (TTFT)
        if first_token_time is None:
            first_token_time = time.perf_counter()
        
        # === Check EOS ===
        if next_token == eos_id:
            break
        
        generated_ids.append(next_token)
        
        # === Decode and stream (properly handles Ġ) ===
        # Decode full sequence to get correct spacing
        decoded_so_far = tokenizer.decode(generated_ids)
        if len(generated_ids) > 1:
            decoded_prev = tokenizer.decode(generated_ids[:-1])
            new_text = decoded_so_far[len(decoded_prev):]
        else:
            new_text = decoded_so_far
        
        print(new_text, end="", flush=True)
        
        # === Early stop on repetition (safety) ===
        if len(generated_ids) >= 20:
            last_20 = generated_ids[-20:]
            # Check if last 10 tokens repeat
            if last_20[:10] == last_20[10:]:
                print("\n[Stopped: repetition detected]", end="")
                break
    
    # === Timing Stats ===
    end_time = time.perf_counter()
    total_time = end_time - start_time
    num_tokens = len(generated_ids)
    
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
    tps = num_tokens / total_time if total_time > 0 else 0
    
    return {
        "generated_ids": generated_ids,
        "num_tokens": num_tokens,
        "total_time_s": total_time,
        "ttft_ms": ttft,
        "tokens_per_second": tps,
    }


def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Error: Tokenizer not found at {TOKENIZER_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = load_model(CHECKPOINT_PATH)
    
    print("\n" + "=" * 60)
    print("🚀 1B MLA MODEL - PHASE 1 INFERENCE")
    print("=" * 60)
    print("Commands:")
    print("  /temp <value>   - Set temperature (default: 0.8)")
    print("  /topk <value>   - Set top_k (default: 40)")
    print("  /topp <value>   - Set top_p (default: 0.95)")
    print("  /rep <value>    - Set repetition penalty (default: 1.1)")
    print("  /max <value>    - Set max tokens (default: 256)")
    print("  /reset          - Reset to defaults")
    print("  exit            - Quit")
    print("=" * 60 + "\n")

    # Default params
    params = {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_new_tokens": 256,
    }

    while True:
        try:
            prompt = input("\nPrompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not prompt:
            continue
        
        # Handle commands
        if prompt.lower() in ["exit", "quit", "/exit", "/quit"]:
            break
        
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            
            if cmd == "/temp" and len(parts) > 1:
                params["temperature"] = float(parts[1])
                print(f"✓ Temperature set to {params['temperature']}")
            elif cmd == "/topk" and len(parts) > 1:
                params["top_k"] = int(parts[1])
                print(f"✓ Top-K set to {params['top_k']}")
            elif cmd == "/topp" and len(parts) > 1:
                params["top_p"] = float(parts[1])
                print(f"✓ Top-P set to {params['top_p']}")
            elif cmd == "/rep" and len(parts) > 1:
                params["repetition_penalty"] = float(parts[1])
                print(f"✓ Repetition penalty set to {params['repetition_penalty']}")
            elif cmd == "/max" and len(parts) > 1:
                params["max_new_tokens"] = int(parts[1])
                print(f"✓ Max tokens set to {params['max_new_tokens']}")
            elif cmd == "/reset":
                params = {
                    "temperature": 0.8,
                    "top_k": 40,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1,
                    "max_new_tokens": 256,
                }
                print("✓ Reset to defaults")
            elif cmd == "/params":
                print(f"Current params: {params}")
            else:
                print(f"Unknown command: {cmd}")
            continue
        
        print("\nCompletion: ", end="", flush=True)
        
        metrics = generate_with_metrics(
            model,
            tokenizer,
            prompt,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            use_cache=True,
        )
        
        print()
        print("-" * 40)
        print(f"📊 Tokens: {metrics['num_tokens']} | "
              f"TTFT: {metrics['ttft_ms']:.1f}ms | "
              f"TPS: {metrics['tokens_per_second']:.1f} tok/s | "
              f"Total: {metrics['total_time_s']:.2f}s")
        print("-" * 40)


if __name__ == "__main__":
    main()