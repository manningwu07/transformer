#!/usr/bin/env python3
import argparse
import os
import random
import struct
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/raw/corpus.txt")
    parser.add_argument("--out", type=str, default="data/shards/train")
    parser.add_argument("--vocab", type=str, default="data/vocab/tokenizer.json")
    parser.add_argument("--shard_size", type=int, default=100_000_000) # 100M tokens per shard
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer = Tokenizer.from_file(args.vocab)
    
    # We use uint32 to be safe for vocab > 65535
    DTYPE = np.uint32 
    print(f"⚙️  Encoding to {args.out} with {DTYPE.__name__}...")

    # Shard state
    shard_idx = 0
    current_tokens = []
    
    def flush_shard():
        nonlocal shard_idx, current_tokens
        if not current_tokens: return
        
        # Write .bin file (Raw tokens)
        bin_path = f"{args.out}-{shard_idx:03d}.bin"
        arr = np.array(current_tokens, dtype=DTYPE)
        with open(bin_path, "wb") as f:
            f.write(arr.tobytes())
            
        # Write .idx file (Document offsets)
        # For a simple LM pre-training, we usually just pack tokens. 
        # But if you need document boundaries, we'd save them here.
        # For efficiency, we will just save the TOTAL count here for the loader.
        idx_path = f"{args.out}-{shard_idx:03d}.idx"
        with open(idx_path, "w") as f:
            f.write(str(len(current_tokens)))
            
        print(f"📦 Saved shard {shard_idx}: {len(current_tokens)/1e6:.1f}M tokens")
        shard_idx += 1
        current_tokens = []

    # Read & Encode
    with open(args.corpus, "r", encoding="utf-8") as f:
        # Use a buffer to batch tokenization calls
        batch = []
        for line in tqdm(f, desc="Tokenizing"):
            text = line.strip()
            if not text: continue
            batch.append(text)
            
            if len(batch) >= 1000:
                encodings = tokenizer.encode_batch(batch)
                for enc in encodings:
                    # Add EOS token (id 2 usually, check your vocab)
                    ids = enc.ids + [2] 
                    current_tokens.extend(ids)
                batch = []
                
                if len(current_tokens) >= args.shard_size:
                    flush_shard()
        
        # Final flush
        if batch:
            encodings = tokenizer.encode_batch(batch)
            for enc in encodings:
                current_tokens.extend(enc.ids + [2])
        flush_shard()

if __name__ == "__main__":
    main()