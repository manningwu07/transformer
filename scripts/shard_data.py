import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

def stream_to_shards(input_path, tokenizer_path, out_dir, shard_size=100_000_000):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the tokenizer you trained
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<eos>")

    print(f"🚀 Sharding {input_path}...")
    
    token_buffer = []
    shard_count = 0
    
    # Process line by line to keep RAM usage low on your Mac
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            text = line.strip()
            if not text:
                continue
            
            # Tokenize and append EOS
            ids = tokenizer.encode(text).ids
            ids.append(eos_id)
            token_buffer.extend(ids)
            
            # Write shard once buffer hits the limit
            if len(token_buffer) >= shard_size:
                out_path = os.path.join(out_dir, f"phase1-train-{shard_count:03d}.bin")
                # uint16 is vital for 16GB VRAM cards (saves IO bandwidth)
                np.array(token_buffer, dtype=np.uint16).tofile(out_path)
                token_buffer = []
                shard_count += 1

    # Final flush
    if token_buffer:
        out_path = os.path.join(out_dir, f"train-{shard_count:03d}.bin")
        np.array(token_buffer, dtype=np.uint16).tofile(out_path)
        
    print(f"✅ Created {shard_count + 1} shards in {out_dir}")

if __name__ == "__main__":
    stream_to_shards(
        input_path="data/raw/phase1_corpus.txt",
        tokenizer_path="data/json/tokenizer.json",
        out_dir="data/shards/phase1"
    )