import os
import glob
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# --- CONFIG ---
RAW_DATA_PATH = "data/raw/corpus.txt" # Output from getData.py
OUTPUT_DIR = "data/binary/train"
TOKENIZER_PATH = "data/tokenizer.json"
# ----------------

def prepare():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"⚙️ Loading Tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # We use uint16 (max 65535). Ensure vocab size fits!
    if tokenizer.get_vocab_size() > 65535:
        print("⚠️ Warning: Vocab size > 65535. Switching to uint32 (2x storage).")
        dtype = np.uint32
    else:
        dtype = np.uint16

    print(f"📖 Reading {RAW_DATA_PATH}...")
    
    # Output binary file
    bin_file = os.path.join(OUTPUT_DIR, "data.bin")
    
    # We write to a temporary file then rename
    tokens_buffer = []
    buffer_size = 100_000_000 # Flush every 100M tokens
    
    total_tokens = 0
    
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f, open(bin_file, "wb") as out_f:
        pbar = tqdm(total=os.path.getsize(RAW_DATA_PATH), unit="B", unit_scale=True)
        
        for line in f:
            pbar.update(len(line.encode('utf-8')))
            line = line.strip()
            if not line: continue
            
            # Encode
            encoded = tokenizer.encode(line).ids
            
            # Add EOS token (Assume ID 2, or check your tokenizer)
            # This allows "packing" multiple short samples into one sequence
            encoded.append(tokenizer.token_to_id("<|endoftext|>") or 0)
            
            tokens_buffer.extend(encoded)
            
            if len(tokens_buffer) >= buffer_size:
                arr = np.array(tokens_buffer, dtype=dtype)
                out_f.write(arr.tobytes())
                total_tokens += len(tokens_buffer)
                tokens_buffer = [] # Clear RAM
        
        # Write remaining
        if tokens_buffer:
            arr = np.array(tokens_buffer, dtype=dtype)
            out_f.write(arr.tobytes())
            total_tokens += len(tokens_buffer)
            
    print(f"✅ Done. Saved {total_tokens} tokens to {bin_file}")

if __name__ == "__main__":
    prepare()