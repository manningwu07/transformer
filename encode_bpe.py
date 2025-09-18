import struct
import numpy as np
from tokenizers import Tokenizer

tok_json = "data/test/tokenizer.json"
tokenizer = Tokenizer.from_file(tok_json)

def encode_file_gptstyle(
    in_path,
    prefix,
    seq_len=512,
    max_shard_bytes=2 * 1024 * 1024 * 1024,
):
    """
    GPT-style dataset creation:
    - Concatenate all text lines with <doc>/<eos> markers
    - Tokenize once
    - Slice into exact fixed-length (seq_len) blocks
    - Write into .bin/.idx shards (like Megatron, GPT-2)
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("data/test/tokenizer.json")
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")

    shard_id = 0
    cur_bytes = 0
    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024*1024)
    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024*1024)

    with open(in_path, "r", encoding="utf-8") as f:
        buffer = []
        for raw in f:
            line = raw.strip()
            if not line: 
                continue
            buffer.append(line)
            if len(buffer) >= 8192:
                cur_bytes = _flush(buffer, tok, data_bin, data_idx, seq_len, cur_bytes, bos_id, eos_id)
                buffer.clear()
                if cur_bytes >= max_shard_bytes:
                    data_bin.close(); data_idx.close()
                    shard_id += 1; cur_bytes = 0
                    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024*1024)
                    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024*1024)

        if buffer:
            cur_bytes = _flush(buffer, tok, data_bin, data_idx, seq_len, cur_bytes, bos_id, eos_id)

    data_bin.close()
    data_idx.close()

def _flush(lines, tok, data_bin, data_idx, seq_len, cur_bytes, bos_id, eos_id):
    encs = tok.encode_batch(lines)
    for enc in encs:
        ids = [bos_id] + enc.ids + [eos_id]
        # break into seq_len windows if needed
        for i in range(0, len(ids)-1, seq_len):
            window = ids[i:i+seq_len]
            start = cur_bytes
            length = len(window)
            data_idx.write(struct.pack("<QQ", start, length))
            arr = np.asarray(window, dtype=np.uint32)
            data_bin.write(arr.tobytes())
            cur_bytes += 4 * length
    return cur_bytes

if __name__ == "__main__":
    encode_file_gptstyle("data/raw/wiki_train.txt", "data/test/wiki_train_ids", seq_len=512)
    print("Done with wiki train (gpt-style)")
    encode_file_gptstyle("data/raw/wiki_val.txt", "data/test/wiki_val_ids", seq_len=512)
    print("Done with wiki val (gpt-style)")
    encode_file_gptstyle("data/raw/wiki_eval.txt", "data/test/wiki_eval_ids", seq_len=512)
    print("Done with wiki eval (gpt-style)")