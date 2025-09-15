import os
import struct
import numpy as np
from tokenizers import Tokenizer

tok_json = "data/test/tokenizer.json"
tokenizer = Tokenizer.from_file(tok_json)

def encode_file(
    in_path,
    prefix,
    max_shard_bytes=2 * 1024 * 1024 * 1024,
    batch_size=8192,
):
    """
    Write .bin + .idx shards given an input .txt file.
    Same binary format as Go ExportTokenIDsBinary.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    shard_id = 0
    cur_bytes = 0
    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024 * 1024)
    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024 * 1024)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    assert bos_id is not None and eos_id is not None, "Missing BOS/EOS in tokenizer"
    
    with open(in_path, "r", encoding="utf-8") as f:
        buffer = []
        for raw in f:
            # Preserve leading/trailing spaces inside the line; drop only newline
            line = raw.rstrip("\n")
            if line == "":
                continue  # skip empty examples to avoid degenerate sequences
            buffer.append(line)
            if len(buffer) >= batch_size:
                cur_bytes = _flush_batch(buffer, tokenizer, data_bin, data_idx, cur_bytes, bos_id, eos_id)
                buffer.clear()
                if cur_bytes >= max_shard_bytes:
                    data_bin.close(); data_idx.close()
                    shard_id += 1; cur_bytes = 0
                    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024*1024)
                    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024*1024)

        if buffer:
            cur_bytes = _flush_batch(buffer, tokenizer, data_bin, data_idx, cur_bytes, bos_id, eos_id)
            buffer.clear()

    data_bin.close()
    data_idx.close()

def _flush_batch(lines, tok, data_bin, data_idx, cur_bytes, bos_id, eos_id):
    # Batch encode with internal parallelism (rust rayon)
    encs = tok.encode_batch(lines)
    for enc in encs:
        ids = [bos_id] + enc.ids + [eos_id]
        start = cur_bytes
        length = len(ids)
        # idx: two uint64 values
        data_idx.write(struct.pack("<QQ>", start, length))
        # bin: contiguous uint32 array
        arr = np.asarray(ids, dtype=np.uint32)
        data_bin.write(arr.tobytes())
        cur_bytes += 4 * length
    return cur_bytes

if __name__ == "__main__":
    # example usage
    encode_file("data/test/wiki_train.txt", "data/test/wiki_train_ids")
    encode_file("data/test/wiki_val.txt", "data/test/wiki_val_ids")
    encode_file("data/test/wiki_eval.txt", "data/test/wiki_eval_ids")
    print("✅ All text files encoded into .bin/.idx shards")