#!/usr/bin/env python3
import os
import re
import struct
import json
import numpy as np
from tokenizers import Tokenizer

tok_json = "data/test/tokenizer.json"
tokenizer = Tokenizer.from_file(tok_json)

def encode_jsonl(
    in_path,
    prefix,
    max_shard_bytes=2 * 1024 * 1024 * 1024,
    batch_size=8192,
):
    """
    Convert Alpaca-style JSONL into .bin/.idx shards.
    Each record has {"human": "...", "bot": "..."}.
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    shard_id = 0
    cur_bytes = 0
    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024 * 1024)
    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024 * 1024)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    assert bos_id is not None and eos_id is not None, "Missing BOS/EOS in tokenizer"

    buffer = []
    with open(in_path, "r", encoding="utf-8") as f:
        for raw in f:
            obj = json.loads(raw)
            human, bot = obj["human"].strip(), obj["bot"].strip()
            
            human = re.sub(r"\s+", " ", human)
            bot   = re.sub(r"\s+", " ", bot)
            
            # Build a single training string
            text = f"<bos>User: {human} <NL> Assistant: {bot}<eos>"
            text = re.sub(r"\s+", " ", text).strip()
            buffer.append(text)

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

    data_bin.close()
    data_idx.close()
    print(f"✅ Finished encoding {in_path} → {prefix}-???.bin/.idx")


def _flush_batch(lines, tok, data_bin, data_idx, cur_bytes, bos_id, eos_id):
    encs = tok.encode_batch(lines)
    for enc in encs:
        ids = [bos_id] + enc.ids + [eos_id]
        start = cur_bytes
        length = len(ids)
        data_idx.write(struct.pack("<QQ", start, length))
        arr = np.asarray(ids, dtype=np.uint32)
        data_bin.write(arr.tobytes())
        cur_bytes += 4 * length
    return cur_bytes


if __name__ == "__main__":
    encode_jsonl("data/raw/alpaca_train.jsonl", "data/test/alpaca_train_ids")
    encode_jsonl("data/raw/alpaca_eval.jsonl", "data/test/alpaca_eval_ids")