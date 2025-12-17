#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from transformer import LLM
from params import Config

# --------------------
# Dataset utilities (Pure Python/Numpy)
# --------------------

class DatasetIterator:
    """
    Standard Python iterator to replace torch.utils.data.IterableDataset
    Yields batches of mlx.arrays.
    """
    def __init__(self, prefix, seq_len, batch_size, repeat=True, shuffle=True, pad_id=0, seed=42):
        self.prefix = prefix
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.shards = sorted(glob.glob(prefix + "-*.idx"))
        assert self.shards, f"No shards found for {prefix}"
        self.rng = random.Random(seed)
        
    def __iter__(self):
        while True:
            # 1. Select Shard
            shard_path = self.rng.choice(self.shards) if self.shuffle else self.shards[0]
            bin_path = shard_path.replace(".idx", ".bin")
            
            idx_arr = np.fromfile(shard_path, dtype=np.uint64).reshape(-1, 2)
            data = np.memmap(bin_path, dtype=np.uint32, mode="r")
            
            order = list(range(len(idx_arr)))
            if self.shuffle:
                self.rng.shuffle(order)
            
            batch_acc_x = []
            batch_acc_y = []

            for i in order:
                st, ln = idx_arr[i]
                arr = data[st // 4 : st // 4 + ln].astype(np.int64)
                
                # Logic from original dataset to slice/pad
                if len(arr) < 2: continue
                
                if len(arr) > self.seq_len + 1:
                    start_off = self.rng.randint(0, len(arr) - (self.seq_len + 1)) if self.shuffle else 0
                    arr = arr[start_off : start_off + self.seq_len + 1]
                elif len(arr) < self.seq_len + 1:
                    pad = np.full((self.seq_len + 1 - len(arr),), self.pad_id, dtype=np.int64)
                    arr = np.concatenate([arr, pad])

                x = arr[:-1]
                y = arr[1:]
                
                batch_acc_x.append(x)
                batch_acc_y.append(y)
                
                if len(batch_acc_x) == self.batch_size:
                    # Convert to MLX array and yield
                    yield mx.array(np.stack(batch_acc_x)), mx.array(np.stack(batch_acc_y))
                    batch_acc_x, batch_acc_y = [], []
            
            # Handle remaining
            if batch_acc_x and not self.repeat:
                 yield mx.array(np.stack(batch_acc_x)), mx.array(np.stack(batch_acc_y))

            if not self.repeat:
                break


# --------------------
# Training Funcs
# --------------------

def loss_fn(model, x, y, pad_id):
    # Mask creation if needed (auto-handled by nn.MultiHeadAttention usually, but explicit here for compile)
    L = x.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
    
    logits, _ = model(x, mask=mask)
    
    # Flatten for Cross Entropy
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    
    # MLX cross_entropy requires ignore_index manually masked usually or use built-in
    # nn.losses.cross_entropy does not have ignore_index in all versions, 
    # so we multiply loss by mask.
    loss = nn.losses.cross_entropy(logits, y)
    
    # Mask padding
    mask_pad = (y != pad_id)
    loss = (loss * mask_pad).sum() / mask_pad.sum()
    return loss

# State update function (Compiled)
@mx.compile
def step(model, optimizer, x, y, pad_id):
    loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y, pad_id)
    optimizer.update(model, grads)
    return loss

# --------------------
# Main
# --------------------

def main(args):
    # Load Vocab
    with open(args.vocab, "r") as f:
        vocab = json.load(f)
    tok2id = vocab["TokenToID"]
    vocab_size = len(tok2id)
    pad_id = tok2id["<pad>"]
    
    print(f"MLX Device: {mx.default_device()}")

    # Initialize Model
    model = LLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        dropout=Config.dropout,
        max_len=Config.max_len
    )
    mx.eval(model.parameters()) # Initialize weights
    print(f"Model params: {sum(x.size for _, x in tree_flatten(model.parameters())) / 1e6:.2f}M")

    # Optimizer: Adafactor as requested
    # Note: MLX Adafactor is available in mlx.optimizers
    optimizer = optim.Adafactor(
        learning_rate=Config.startLr,
        beta1=Config.beta1,  # Adafactor usually doesn't use standard betas the same way, but MLX API aligns often
        weight_decay=Config.weight_decay
    )

    # Resume?
    if os.path.exists(args.resumePath):
        print(f"Loading checkpoint {args.resumePath}")
        model.load_weights(args.resumePath)

    # Dataset
    train_iter = DatasetIterator(args.train, Config.seq_len, Config.batch_size, pad_id=pad_id)
    
    # Loop
    steps = 0
    t0 = time.time()
    total_loss = 0
    
    try:
        model.train()
        for x, y in train_iter:
            loss = step(model, optimizer, x, y, pad_id)
            mx.eval(loss) # Force computation
            
            total_loss += loss.item()
            steps += 1
            
            if steps % 10 == 0:
                dt = time.time() - t0
                tps = (Config.batch_size * Config.seq_len * 10) / dt
                print(f"Step {steps} | Loss: {total_loss/10:.4f} | TPS: {tps:.0f}")
                total_loss = 0
                t0 = time.time()
                
            if steps % args.eval_every_steps == 0:
                print("Saving model...")
                model.save_weights("models/best_model.safetensors") # Use safetensors for MLX
                
            if steps >= args.max_steps:
                break
                
    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        model.save_weights("models/last_run.safetensors")

def tree_flatten(tree):
    # Helper to count params
    import mlx.utils
    return mlx.utils.tree_flatten(tree)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=Config.totalOptSteps)
    parser.add_argument("--eval_every_steps", type=int, default=Config.eval_every_steps)
    parser.add_argument("--resumePath", type=str, default="models/best_model.safetensors")
    args = parser.parse_args()

    main(args)