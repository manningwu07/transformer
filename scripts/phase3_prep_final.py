#!/usr/bin/env python3
"""
Prepares Phase 3 data into standard train/val structure.

Input:
  data/shards/phase3/longctx/*.bin
  data/shards/phase3/sft/*.bin
  data/shards/phase3/dpo/*.npz (kept separate)

Output:
  data/shards/phase3_final/train/*.bin  (interleaved longctx + sft)
  data/shards/phase3_final/val/*.bin    (held-out validation)
  data/shards/phase3_final/dpo/*.npz    (copied as-is)
"""
import argparse
import glob
import os
import random
import shutil
from typing import List

import numpy as np
from tqdm import tqdm


def collect_all_tokens(shard_dirs: List[str], pattern: str, dtype: np.dtype) -> np.ndarray:
    """Load all tokens from multiple directories into one array."""
    all_tokens = []
    for d in shard_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, pattern))):
            mm = np.memmap(f, dtype=dtype, mode="r")
            all_tokens.append(np.array(mm))
    
    if not all_tokens:
        raise FileNotFoundError(f"No shards found in {shard_dirs}")
    
    return np.concatenate(all_tokens)


def write_shards(
    tokens: np.ndarray,
    out_dir: str,
    prefix: str,
    shard_size: int,
    dtype: np.dtype,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for start in range(0, len(tokens), shard_size):
        chunk = tokens[start : start + shard_size]
        path = os.path.join(out_dir, f"{prefix}-{idx:05d}.bin")
        chunk.astype(dtype).tofile(path)
        idx += 1
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3_dir", type=str, default="data/shards/phase3")
    ap.add_argument("--out_dir", type=str, default="data/shards/phase3_final")
    ap.add_argument("--val_ratio", type=float, default=0.02, help="Fraction for validation")
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", type=str, default="uint16")
    
    # Mixing weights for train set
    ap.add_argument("--longctx_weight", type=float, default=0.6)
    ap.add_argument("--sft_weight", type=float, default=0.4)
    
    args = ap.parse_args()
    
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    rng = np.random.default_rng(args.seed)
    
    longctx_dir = os.path.join(args.phase3_dir, "longctx")
    sft_dir = os.path.join(args.phase3_dir, "sft")
    dpo_dir = os.path.join(args.phase3_dir, "dpo")
    
    print("📥 Loading longctx tokens...")
    longctx_tokens = collect_all_tokens([longctx_dir], "*.bin", dtype)
    print(f"   Loaded {len(longctx_tokens):,} longctx tokens")
    
    print("📥 Loading SFT tokens...")
    sft_tokens = collect_all_tokens([sft_dir], "*.bin", dtype)
    print(f"   Loaded {len(sft_tokens):,} SFT tokens")
    
    # Interleave by weight (sample proportionally)
    total_weight = args.longctx_weight + args.sft_weight
    longctx_ratio = args.longctx_weight / total_weight
    
    # Determine final sizes based on smaller source scaled by weight
    longctx_target = int(len(longctx_tokens))
    sft_target = int(longctx_target * (args.sft_weight / args.longctx_weight))
    sft_target = min(sft_target, len(sft_tokens))
    
    # Subsample SFT if needed
    if sft_target < len(sft_tokens):
        indices = rng.choice(len(sft_tokens), size=sft_target, replace=False)
        indices.sort()
        sft_tokens = sft_tokens[indices]
    
    print(f"🔀 Interleaving: {len(longctx_tokens):,} longctx + {len(sft_tokens):,} sft")
    
    # Simple interleave: chunk and alternate
    chunk_size = 10_000_000  # 10M tokens per chunk
    mixed = []
    
    lc_chunks = [longctx_tokens[i:i+chunk_size] for i in range(0, len(longctx_tokens), chunk_size)]
    sft_chunks = [sft_tokens[i:i+chunk_size] for i in range(0, len(sft_tokens), chunk_size)]
    
    # Weighted interleave
    lc_idx, sft_idx = 0, 0
    while lc_idx < len(lc_chunks) or sft_idx < len(sft_chunks):
        if lc_idx < len(lc_chunks) and (sft_idx >= len(sft_chunks) or rng.random() < longctx_ratio):
            mixed.append(lc_chunks[lc_idx])
            lc_idx += 1
        elif sft_idx < len(sft_chunks):
            mixed.append(sft_chunks[sft_idx])
            sft_idx += 1
    
    all_tokens = np.concatenate(mixed)
    
    # Shuffle at block level for better mixing
    block_size = args.shard_size
    n_blocks = len(all_tokens) // block_size
    blocks = [all_tokens[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    remainder = all_tokens[n_blocks*block_size:]
    
    rng.shuffle(blocks)
    all_tokens = np.concatenate(blocks + ([remainder] if len(remainder) > 0 else []))
    
    # Split train/val
    val_size = int(len(all_tokens) * args.val_ratio)
    val_size = (val_size // block_size) * block_size  # Align to block boundary
    
    val_tokens = all_tokens[:val_size]
    train_tokens = all_tokens[val_size:]
    
    print(f"📊 Split: {len(train_tokens):,} train / {len(val_tokens):,} val")
    
    # Write shards
    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    
    print("💾 Writing train shards...")
    n_train = write_shards(train_tokens, train_dir, "phase3-train", args.shard_size, dtype)
    
    print("💾 Writing val shards...")
    n_val = write_shards(val_tokens, val_dir, "phase3-val", args.shard_size, dtype)
    
    # Copy DPO files
    dpo_out = os.path.join(args.out_dir, "dpo")
    if os.path.isdir(dpo_dir):
        print("📋 Copying DPO shards...")
        os.makedirs(dpo_out, exist_ok=True)
        for f in glob.glob(os.path.join(dpo_dir, "*.npz")):
            shutil.copy2(f, dpo_out)
    
    print(f"""
✅ Phase 3 preparation complete:
   Train: {train_dir} ({n_train} shards, {len(train_tokens):,} tokens)
   Val:   {val_dir} ({n_val} shards, {len(val_tokens):,} tokens)
   DPO:   {dpo_out}
""")


if __name__ == "__main__":
    main()