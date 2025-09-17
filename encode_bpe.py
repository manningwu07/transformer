import os
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    doc_id = tokenizer.token_to_id("<doc>") if tokenizer.token_to_id("<doc>") is not None else eos_id
    assert bos_id is not None and eos_id is not None

    # Collect text corpus into one long string
    texts = []
    with open(in_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # prepend <doc> per line/article (since build_wiki already chunks per article)
            texts.append("<doc> " + line)

    full_text = "\n".join(texts)
    enc = tokenizer.encode(full_text)
    ids = [bos_id] + enc.ids + [eos_id]

    # Now slice into fixed seq_len blocks
    shard_id = 0
    cur_bytes = 0
    data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024*1024)
    data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024*1024)

    for i in range(0, len(ids) - seq_len, seq_len):
        window = ids[i:i+seq_len]
        start = cur_bytes
        length = len(window)
        data_idx.write(struct.pack("<QQ", start, length))
        arr = np.asarray(window, dtype=np.uint32)
        data_bin.write(arr.tobytes())
        cur_bytes += 4 * length
        if cur_bytes >= max_shard_bytes:
            data_bin.close(); data_idx.close()
            shard_id += 1; cur_bytes = 0
            data_bin = open(f"{prefix}-{shard_id:03d}.bin", "wb", buffering=1024*1024)
            data_idx = open(f"{prefix}-{shard_id:03d}.idx", "wb", buffering=1024*1024)

    data_bin.close()
    data_idx.close()


if __name__ == "__main__":
    encode_file_gptstyle("data/test/wiki_train.txt", "data/test/wiki_train_ids", seq_len=512)
    print("Done with wiki train (gpt-style)")
    encode_file_gptstyle("data/test/wiki_val.txt", "data/test/wiki_val_ids", seq_len=512)
    print("Done with wiki val (gpt-style)")
    encode_file_gptstyle("data/test/wiki_eval.txt", "data/test/wiki_eval_ids", seq_len=512)
    print("Done with wiki eval (gpt-style)")