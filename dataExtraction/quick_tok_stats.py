#!/usr/bin/env python3
import numpy as np, json
tok = json.load(open("data/json/vocab.json"))
I2T = tok["IDToToken"]

def top_ids(prefix, n=1_000_000):
    dat = np.memmap(f"{prefix}-000.bin", dtype=np.uint32, mode="r")
    a = np.array(dat[: min(n, len(dat))])
    ids, cnt = np.unique(a, return_counts=True)
    top = sorted(zip(cnt, ids), reverse=True)[:40]
    for c, i in top:
        print(f"{i}\t{c}\t{I2T[i] if i < len(I2T) else 'UNK'}")
    print("total scanned:", len(a))

top_ids("data/shards/train")