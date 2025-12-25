import os
import numpy as np
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from tqdm import tqdm

def stream_to_bin():
    # --- CONFIG ---
    tokenizer_path = "data/json/tokenizer_32k.json"
    out_dir = "data/shards/phase1"
    os.makedirs(out_dir, exist_ok=True)
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>") or 0
    
    target_tokens = 10_000_000_000 # 10B
    shard_size = 100_000_000 
    
    # 1. Load the three parts of SmollM
    print("🌊 Interleaving SmollM subsets (Cosmopedia + FineWeb + Python)...")
    
    # We use these subsets specifically because they make up the SmollM training set
    cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
    python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)

    # 2. Interleave them (Weighting them equally or by size)
    # This gives you a representative slice of the whole corpus
    ds = interleave_datasets([cosmo, fineweb, python], probabilities=[0.4, 0.5, 0.1], seed=42)

    current_shard_tokens = []
    total_written = 0
    shard_idx = 0

    pbar = tqdm(total=target_tokens, unit="tok")

    for entry in ds:
        text = entry.get("text") or ""
        if not text: continue
        
        tokens = tokenizer.encode(text).ids
        tokens.append(eos_id)
        current_shard_tokens.extend(tokens)
        
        if len(current_shard_tokens) >= shard_size:
            arr = np.array(current_shard_tokens[:shard_size], dtype=np.uint16)
            out_path = os.path.join(out_dir, f"train-{shard_idx:03d}.bin")
            arr.tofile(out_path)
            
            total_written += shard_size
            pbar.update(shard_size)
            current_shard_tokens = current_shard_tokens[shard_size:]
            shard_idx += 1
            
        if total_written >= target_tokens:
            break

    pbar.close()
    print(f"✅ Success. 10B tokens written to {shard_idx} shards.")

if __name__ == "__main__":
    stream_to_bin()