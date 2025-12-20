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
from mlx.utils import tree_map, tree_flatten

from transformer_mlx import LLM
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
                    yield mx.array(np.stack(batch_acc_x)), mx.array(np.stack(batch_acc_y))
                    batch_acc_x, batch_acc_y = [], []
            
            if batch_acc_x and not self.repeat:
                 yield mx.array(np.stack(batch_acc_x)), mx.array(np.stack(batch_acc_y))

            if not self.repeat:
                break


# --------------------
# Training Funcs
# --------------------

def loss_fn(model, x, y, pad_id):
    # Auto-masking for causal attention
    L = x.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
    
    logits, _ = model(x, mask=mask)
    
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    
    # Cross entropy with masking
    loss = nn.losses.cross_entropy(logits, y)
    mask_pad = (y != pad_id)
    loss = (loss * mask_pad).sum() / mask_pad.sum()
    return loss

def clip_grad_norm(grads, max_norm):
    """
    Computes the global norm of the gradients and scales them 
    if the norm exceeds max_norm.
    """
    # Flatten all gradients into a single vector of norms
    total_norm = mx.sqrt(sum(mx.sum(g ** 2) for _, g in tree_flatten(grads)))
    
    # Calculate scaling factor
    # scale = max_norm / (total_norm + 1e-6)
    # if total_norm < max_norm, we want scale = 1.0 (min takes care of this)
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-6))
    
    # Apply scale
    return tree_map(lambda g: g * scale, grads), total_norm

@mx.compile
def step(model, optimizer, x, y, pad_id, max_grad_norm):
    loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y, pad_id)
    
    # Aggressive Clipping
    grads, grad_norm = clip_grad_norm(grads, max_grad_norm)
    
    optimizer.update(model, grads)
    return loss, grad_norm

def to_fp16(model):
    # Helper to cast all floating point parameters to float16
    # This enables the "Fast Path" on M-series chips
    def _cast(p):
        if mx.issubdtype(p.dtype, mx.floating):
            return p.astype(mx.float16)
        return p
    model.update(tree_map(_cast, model.parameters()))

# --------------------
# Main
# --------------------

def main(args):
    with open(args.vocab, "r") as f:
        vocab = json.load(f)
    tok2id = vocab["TokenToID"]
    vocab_size = len(tok2id)
    pad_id = tok2id["<pad>"]
    
    print(f"MLX Device: {mx.default_device()}")

    # 1. Init Model
    model = LLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        dropout=Config.dropout,
        max_len=Config.max_len
    )
    mx.eval(model.parameters()) 

    # 2. SPEED: Cast to Float16
    # If loss goes NaN, this is the first thing to check.
    # But for "x10 speed", this is required on Mac.
    print("🚀 Casting model to float16 for max speed...")
    to_fp16(model)

    print(f"Model params: {sum(x.size for _, x in tree_flatten(model.parameters())) / 1e6:.2f}M")

    # 3. Optimizer
    # Adafactor is robust, good choice for lower precision
    optimizer = optim.Adafactor(
        learning_rate=Config.startLr,
        beta1=Config.beta1,
        weight_decay=Config.weight_decay,
        eps=1e-16 # stabilizing epsilon for float16
    )

    if os.path.exists(args.resumePath):
        print(f"Loading checkpoint {args.resumePath}")
        model.load_weights(args.resumePath)

    train_iter = DatasetIterator(args.train, Config.seq_len, Config.batch_size, pad_id=pad_id)
    
    steps = 0
    t0 = time.time()
    total_loss = 0
    
    # Clip value from Config
    grad_clip = getattr(Config, "grad_clip", 1.0)
    print(f"✂️  Aggressive Gradient Clipping Enabled (max_norm={grad_clip})")

    try:
        model.train()
        for x, y in train_iter:
            # 4. Step with Clipping
            loss, g_norm = step(model, optimizer, x, y, pad_id, grad_clip)
            mx.eval(loss, g_norm) # Force sync
            
            # Check for NaN immediately
            if math.isnan(loss.item()):
                print(f"❌ NaN detected at step {steps}. Gradient Norm was: {g_norm.item()}")
                print("Tip: Lower learning rate or switch back to bfloat16 if persistent.")
                # Optional: Skip update or break? 
                # usually breaking is better so you don't poison the weights
                break

            total_loss += loss.item()
            steps += 1
            
            if steps % 10 == 0:
                dt = time.time() - t0
                tps = (Config.batch_size * Config.seq_len * 10) / dt
                print(f"Step {steps} | Loss: {total_loss/10:.4f} | GNorm: {g_norm.item():.2f} | TPS: {tps:.0f}")
                total_loss = 0
                t0 = time.time()
                
            if steps % args.eval_every_steps == 0:
                print("Saving model...")
                model.save_weights("models/best_model.safetensors")
                
            if steps >= args.max_steps:
                break
                
    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        model.save_weights("models/last_run.safetensors")

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