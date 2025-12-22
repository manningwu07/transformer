import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

def stream_conversion(txt_path, out_dir, tokenizer_path):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    
    shard_size_tokens = 100_000_000 
    batch_size_lines = 50000 
    
    token_buffer = []
    shard_count = 0
    line_buffer = []

    def flush_lines(lines):
        encodings = tokenizer.encode_batch(lines)
        ids = []
        for enc in encodings:
            ids.extend(enc.ids)
            ids.append(eos_id)
        return ids

    file_size = os.path.getsize(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=file_size, unit='B', unit_scale=True)
        for line in f:
            line_buffer.append(line)
            pbar.update(len(line.encode('utf-8')))
            if len(line_buffer) >= batch_size_lines:
                token_buffer.extend(flush_lines(line_buffer))
                line_buffer = []
                if len(token_buffer) >= shard_size_tokens:
                    out_path = os.path.join(out_dir, f"train-{shard_count:03d}.bin")
                    # USE UINT16 TO SAVE 50% DISK SPACE AND IO BANDWIDTH
                    np.array(token_buffer, dtype=np.uint16).tofile(out_path)
                    shard_count += 1
                    token_buffer = []

        if line_buffer:
            token_buffer.extend(flush_lines(line_buffer))
        if token_buffer:
            out_path = os.path.join(out_dir, f"train-{shard_count:03d}.bin")
            np.array(token_buffer, dtype=np.uint16).tofile(out_path)

if __name__ == "__main__":
    stream_conversion("data/raw/phase1_corpus.txt", "data/shards", "data/json/tokenizer.json")