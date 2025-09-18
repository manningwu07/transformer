#!/usr/bin/env python3
"""
Split a big training text file into train+val subsets.

Input:
  data/test/wiki_train.txt  (or whatever your "train" text is)

Output:
  data/test/wiki_train.txt      (reduced train subset)
  data/test/wiki_val.txt        (held-out validation subset)

By default: 90% train, 10% val
"""

import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True, help="Input training .txt file")
    parser.add_argument("--train-out", type=str, required=True, help="Output path for new train.txt")
    parser.add_argument("--val-out", type=str, required=True, help="Output path for new val.txt")
    parser.add_argument("--val-frac", type=float, default=0.03, help="Fraction for validation split (default=0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # read all lines
    print(f"Loading {args.infile} ...")
    with open(args.infile, "r", encoding="utf-8") as f:
        docs = f.readlines()

    print(f"Total docs: {len(docs)}")

    # shuffle
    random.shuffle(docs)

    n_val = int(len(docs) * args.val_frac)
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        f.writelines(train_docs)
    with open(args.val_out, "w", encoding="utf-8") as f:
        f.writelines(val_docs)

    print(f"Wrote {len(train_docs)} → {args.train_out}")
    print(f"Wrote {len(val_docs)} → {args.val_out}")


if __name__ == "__main__":
    main()