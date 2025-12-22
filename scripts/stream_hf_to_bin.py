import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

def stream_to_bin(dataset_name, tokenizer_path, out_dir, subset=None):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    
    # 1. Load dataset with streaming=True (Zero disk usage for text)
    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    
    shard_size = 100_000_000
    token_buffer = []
    shard_count = 0

    print(f"🌊 Streaming {dataset_name} directly to binary shards...")

    for entry in tqdm(ds):
        # HF datasets usually use "text" or "content" keys
        text = entry.get("text") or entry.get("content") or ""
        if not text: continue
        
        # 2. Tokenize in RAM
        ids = tokenizer.encode(text).ids
        ids.append(eos_id)
        token_buffer.extend(ids)
        
        # 3. Write to Disk only when buffer is full
        if len(token_buffer) >= shard_size:
            out_path = os.path.join(out_dir, f"phase2-{shard_count:03d}.bin")
            np.array(token_buffer, dtype=np.uint32).tofile(out_path)
            token_buffer = []
            shard_count += 1

    # Final flush
    if token_buffer:
        out_path = os.path.join(out_dir, f"phase2-{shard_count:03d}.bin")
        np.array(token_buffer, dtype=np.uint32).tofile(out_path)

if __name__ == "__main__":
    # Example for Phase 2 (Math/Code)
    stream_to_bin(
        dataset_name="HuggingFaceTB/finemath", 
        tokenizer_path="data/json/tokenizer.json", 
        out_dir="data/shards/phase2",
        subset="finemath-4plus"
    )