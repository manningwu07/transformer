#!/usr/bin/env python3
import argparse
import io
import os
import random
import struct
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import re

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
    ap.add_argument("--corpus", type=str, default="data/raw/blended_corpus.txt")
    ap.add_argument("--tokenizer", type=str, default="data/json/tokenizer.json")
    ap.add_argument("--outdir", type=str, default="data/shards")
    # Optional cap: if >0, truncate long examples; if 0, keep full length
    ap.add_argument("--seq_len", type=int, default=0)
    ap.add_argument("--max_shard_bytes", type=int, default=1 * 1024 * 1024 * 1024) # 1GB
    ap.add_argument("--splits", type=str, default="0.90,0.05,0.05")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--truncate_policy", type=str, default="truncate",
                    choices=["truncate","drop"],
                    help="When seq_len>0 and example is longer: truncate or drop")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")
    assert bos_id is not None and eos_id is not None, "Tokenizer missing <bos>/<eos>"
    PAD_ID = tok.token_to_id("<pad>")
    if PAD_ID is None:
        raise ValueError("Tokenizer missing <pad> (required for later collate)")

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
    lengths = {k: [] for k in writers}

    END_TAGS = set([
        "</c4>", "</oscar>", "</dialog>", "</code>", "</stack>",
        "</math>", "</mathqa>", "</comp_math>",
        "[/c4]", "[/oscar]", "[/dialog]", "[/code]", "[/stack]",
        "[/math]", "[/mathqa]", "[/comp_math]"
    ])

    with open(args.corpus, "r", encoding="utf-8") as f:
        buf = io.StringIO()
        batch = []
        for line in tqdm(f, desc="Encoding → shards", mininterval=3.0):
            # accumulate until end tag
            buf.write(line)
            if line.strip() in END_TAGS:
                text = buf.getvalue().strip()
                buf.seek(0)
                buf.truncate(0)
                if not text:
                    continue

                batch.append(text)

            # encode & write in mini-batches to save tokenizer calls
            if len(batch) >= 32:  # tune batch size
                encs = tok.encode_batch(batch)
                batch.clear()
                for enc in encs:
                    full_ids = [bos_id] + enc.ids + [eos_id]
                    if len(full_ids) < 4 or PAD_ID in full_ids:
                        continue
                    # pick split
                    p = random.random()
                    split = (
                        "train" if p < r_train else
                        "val"   if p < r_train + r_val else
                        "test"
                    )
                    if args.seq_len and len(full_ids) > args.seq_len:
                        if args.truncate_policy == "truncate":
                            full_ids = full_ids[:args.seq_len]
                        else:
                            continue
                    w = writers[split]
                    if w["cur"] >= args.max_shard_bytes:
                        next_w(w)
                    write_seq(w, full_ids)
                    counts[split] += 1
                    lengths[split].append(len(full_ids))

        # flush any remaining texts
        if batch:
            for enc in tok.encode_batch(batch):
                full_ids = [bos_id] + enc.ids + [eos_id]
                if len(full_ids) < 4 or PAD_ID in full_ids:
                    continue
                write_seq(writers["train"], full_ids)
                counts["train"] += 1
                lengths["train"].append(len(full_ids))

    for s in writers:
        w = writers[s]
        if w["bin"]:
            w["bin"].close()
            w["idx"].close()

    print("✅ Done. Sequences:", counts)
    # Report stats (mean/std) per split
    import math
    def stat(xs):
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(xs, dtype=np.float64)
        return (arr.mean(), arr.std(ddof=0), np.median(arr), np.percentile(arr, 90))
    for split in ("train","val","test"):
        mu, sd, p50, p90 = stat(lengths[split])
        print(f"[{split}] tokens: mean={mu:.1f}, std={sd:.1f}, p50={p50:.1f}, p90={p90:.1f}, n={len(lengths[split])}")
    print(f"Shards at: {args.outdir}")

if __name__ == "__main__":
    main()