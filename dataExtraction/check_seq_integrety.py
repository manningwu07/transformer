# check_seq_integrity.py
import numpy as np, json
tok = json.load(open("data/json/vocab.json"))
T2I = tok["TokenToID"]; I2T = tok["IDToToken"]
bos, eos, pad = T2I["<bos>"], T2I["<eos>"], T2I["<pad>"]

def scan(prefix):
    idx = np.fromfile(f"{prefix}-000.idx", dtype=np.uint64).reshape(-1,2)
    dat = np.memmap(f"{prefix}-000.bin", dtype=np.uint32, mode="r")
    bad = 0
    for j,(start, ln) in enumerate(idx[:2000]):  # sample first 2k
        arr = dat[start//4:start//4+ln]
        if len(arr)<2: continue
        if arr[0] != bos or arr[-1] != eos:
            bad += 1
        # no mid-BOS/EOS
        if (arr[1:-1]==bos).any() or (arr[1:-1]==eos).any():
            bad += 1
        # no pad between tokens (pads only appear when collating to fixed len; shards shouldn’t contain pad)
        if (arr==pad).any():
            bad += 1
    print(prefix, "bad seqs:", bad, "of", len(idx[:2000]))
scan("data/shards/train")
scan("data/shards/val")
scan("data/shards/test")