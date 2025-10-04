#!/usr/bin/env python3
import argparse
import os
import random
import struct
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

def open_w(wdir, split):
    return {"bin": None, "idx": None, "shard_id": 0, "cur": 0, "dir": wdir, "split": split}

def next_w(w):
    if w["bin"]:
        w["bin"].close()
        w["idx"].close()
    base = os.path.join(w["dir"], f"{w['split']}-{w['shard_id']:03d}")
    os.makedirs(w["dir"], exist_ok=True)
    w["bin"] = open(base + ".bin", "wb", buffering=1024 * 1024)
    w["idx"] = open(base + ".idx", "wb", buffering=1024 * 1024)
    w["cur"] = 0
    w["shard_id"] += 1

def write_seq(w, ids):
    start = w["cur"]
    ln = len(ids)
    w["idx"].write(struct.pack("<QQ", start, ln))
    arr = np.asarray(ids, dtype=np.uint32)
    w["bin"].write(arr.tobytes())
    w["cur"] += 4 * ln

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="blended_corpus.txt")
    ap.add_argument("--tokenizer", type=str, default="data/test/tokenizer.json")
    ap.add_argument("--outdir", type=str, default="data/shards")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_shard_bytes", type=int, default=1 * 1024 * 1024 * 1024)
    ap.add_argument("--splits", type=str, default="0.90,0.05,0.05")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")
    assert bos_id is not None and eos_id is not None, "Tokenizer missing <bos>/<eos>"

    r_train, r_val, r_test = [float(x) for x in args.splits.split(",")]
    assert abs(r_train + r_val + r_test - 1.0) < 1e-6

    writers = {
        "train": open_w(args.outdir, "train"),
        "val": open_w(args.outdir, "val"),
        "test": open_w(args.outdir, "test"),
    }
    for s in writers:
        next_w(writers[s])

    counts = {k: 0 for k in writers}

    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Encoding → shards"):
            text = line.strip()
            if not text:
                continue
            enc = tok.encode(text)
            ids = [bos_id] + enc.ids + [eos_id]

            for i in range(0, max(1, len(ids) - 1), args.seq_len):
                window = ids[i : i + args.seq_len]

                p = random.random()
                if p < r_train:
                    split = "train"
                elif p < r_train + r_val:
                    split = "val"
                else:
                    split = "test"

                w = writers[split]
                if w["cur"] >= args.max_shard_bytes:
                    next_w(w)
                write_seq(w, window)
                counts[split] += 1

    for s in writers:
        w = writers[s]
        if w["bin"]:
            w["bin"].close()
            w["idx"].close()

    print("✅ Done. Sequences:", counts)
    print(f"Shards at: {args.outdir}")

if __name__ == "__main__":
    main()