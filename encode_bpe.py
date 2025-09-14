#!/usr/bin/env python3
from tokenizers import Tokenizer
import struct
import os

tok_json = "data/test/tokenizer.json"
tokenizer = Tokenizer.from_file(tok_json)

def encode_file(in_path, prefix, max_shard_bytes=2*1024*1024*1024):
    """
    Write .bin + .idx shards given an input .txt file.
    Same binary format as Go ExportTokenIDsBinary.
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    shard_id = 0
    cur_bytes = 0
    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb")
    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb")

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Encode with BPE
            enc = tokenizer.encode(line)
            ids = [tokenizer.token_to_id("<bos>")] + enc.ids + [tokenizer.token_to_id("<eos>")]

            # compute start offset
            start = cur_bytes
            length = len(ids)

            # write idx: (start, length) as uint64
            data_idx.write(struct.pack("<Q", start))
            data_idx.write(struct.pack("<Q", length))

            # write bin: token IDs as uint32
            for i in ids:
                data_bin.write(struct.pack("<I", i))
            cur_bytes += 4 * length

            # rollover if shard is too big
            if cur_bytes >= max_shard_bytes:
                data_bin.close()
                data_idx.close()
                shard_id += 1
                data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb")
                data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb")
                cur_bytes = 0

    data_bin.close()
    data_idx.close()

if __name__ == "__main__":
    # example usage
    encode_file("data/test/wiki_train.txt", "data/test/wiki_train_ids")
    encode_file("data/test/wiki_val.txt", "data/test/wiki_val_ids")
    encode_file("data/test/wiki_eval.txt", "data/test/wiki_eval_ids")
    print("✅ All text files encoded into .bin/.idx shards")