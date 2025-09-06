#!/usr/bin/env python3
# build_cloze_eval.py
import argparse
import hashlib
import os
import re
from typing import Tuple, Optional

from datasets import load_dataset
import nltk


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def ascii_clean(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ord(ch) < 128 else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Simple extractors. You can add many more patterns later.
PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # X is a/an Y
    (re.compile(r"^([a-z0-9][a-z0-9 .'\-]{1,80}) is an? ([^.,;:()]+)"), "is a"),
    # X was a/an Y
    (re.compile(r"^([a-z0-9][a-z0-9 .'\-]{1,80}) was an? ([^.,;:()]+)"), "was a"),
    # the capital of X is Y
    (re.compile(r"^the capital of ([a-z0-9 .'\-]+) is ([a-z0-9 .'\-]+)"), "the capital of"),
)


def try_extract(sentence: str) -> Optional[Tuple[str, str]]:
    s = ascii_clean(sentence)
    for pat, bridge in PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        if pat.pattern.startswith("^the capital"):
            country, city = m.group(1).strip(), m.group(2).strip()
            if not country or not city:
                continue
            prompt = f"the capital of {country} is"
            completion = city
            return prompt, completion
        subj, pred = m.group(1).strip(), m.group(2).strip()
        if not subj or not pred:
            continue
        prompt = f"{subj} {bridge}"
        completion = pred
        return prompt, completion
    return None


def hash_split(key: str, frac: float, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return "eval" if v < frac else "train"


def main():
    ap = argparse.ArgumentParser("Build cloze eval TSV from Wikipedia")
    ap.add_argument("--snapshot", default="20231101.en")
    ap.add_argument("--out", default="data/test/cloze_eval.tsv")
    ap.add_argument("--eval-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20000,
                    help="max cloze rows to write (0=all)")
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    ensure_punkt()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # truncate
    open(args.out, "w", encoding="utf-8").close()

    ds = load_dataset("wikimedia/wikipedia",
                      args.snapshot, split="train",
                      streaming=args.streaming)

    wrote = 0
    for i, art in enumerate(ds):
        title = art.get("title") or ""
        text = art.get("text") or ""
        if not text or not title:
            continue
        # use only held-out eval articles for cloze
        if hash_split(title, args.eval_frac, args.seed) != "eval":
            continue
        for sent in nltk.sent_tokenize(text):
            res = try_extract(sent)
            if not res:
                continue
            prompt, completion = res
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(f"{prompt}\t{completion}\n")
            wrote += 1
            if args.limit and wrote >= args.limit:
                break
        if args.limit and wrote >= args.limit:
            break
        if (i + 1) % 1000 == 0:
            print(f"scanned {i+1} articles, wrote {wrote} cloze rows")

    print(f"done. wrote {wrote} rows → {args.out}")


if __name__ == "__main__":
    main()