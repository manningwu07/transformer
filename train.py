#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import signal
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from eval import evaluate
from transformer import GPT2LikeLM
from params import Config


# --------------------
# Dataset utilities
# --------------------

class IndexedBinaryDataset(torch.utils.data.IterableDataset):
    """
    Streams examples from .bin/.idx shards produced by ExportTokenIDsBinary.
    Each example = (x, y), where x is input seq, y is target seq.
    """

    def __init__(self, prefix, seq_len, repeat=False, shuffle=False, pad_id=0):
        self.prefix = prefix
        self.seq_len = seq_len
        self.repeat = repeat
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.shards = sorted(glob.glob(prefix + "-*.idx"))
        assert self.shards, f"No shards found for prefix {prefix}"

    def _iter_shard(self, idx_path):
        bin_path = idx_path.replace(".idx", ".bin")
        idx_arr = np.fromfile(idx_path, dtype=np.uint64).reshape(-1, 2)
        data = np.memmap(bin_path, dtype=np.uint32, mode="r")

        order = np.arange(len(idx_arr))
        if self.shuffle:
            np.random.shuffle(order)

        for j in order:
            start, length = idx_arr[j]
            arr = data[start // 4 : start // 4 + length]
            ids = torch.from_numpy(np.array(arr, dtype=np.int64))
            ids = ids[: self.seq_len]
            if len(ids) < self.seq_len:
                pad = torch.zeros(self.seq_len - len(ids), dtype=torch.long)
                ids = torch.cat([ids, pad])
            yield ids[:-1], ids[1:]

    def __iter__(self):
        while True:
            for shard in self.shards:
                yield from self._iter_shard(shard)
            if not self.repeat:
                break


# --------------------
# Vocab utils
# --------------------

def load_vocab(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["TokenToID"], data["IDToToken"]


# --------------------
# Checkpoint helpers
# --------------------

def save_model_state(model, optimizer, scheduler, step, path, msg=""):
    print("Saving model state...")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
        },
        path,
    )
    print(f"💾 Saved {msg} → {path}")


def crash_handler(sig, frame):
    global model, optimizer, scheduler, global_step
    if model is not None:
        os.makedirs("models", exist_ok=True)
        save_model_state(
            model,
            optimizer,
            scheduler,
            global_step,
            "models/curr_model.pt",
            "CURRENT MODEL (crash-save)",
        )
    sys.exit(0)


# --------------------
# Scheduler
# --------------------

def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-8):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------
# Training loop
# --------------------

best_val_loss = float("inf")
model = None
optimizer = None
scheduler = None
global_step = 0


def main(args):
    global model, optimizer, scheduler, best_val_loss, global_step

    # Handle Ctrl-C / SIGTERM
    signal.signal(signal.SIGINT, crash_handler)
    signal.signal(signal.SIGTERM, crash_handler)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Training on: {device}")

    # ---- Load vocab ----
    tok2id, id2tok = load_vocab(args.vocab)
    vocab_size = len(id2tok)
    print(f"Loaded vocab size: {vocab_size}")
    pad_id = tok2id["<pad>"]

    # ---- Datasets ----
    train_ds = IndexedBinaryDataset(args.train, Config.seq_len, shuffle=True,  repeat=True,  pad_id=pad_id)
    val_ds   = IndexedBinaryDataset(args.val,   Config.seq_len, shuffle=False, repeat=False, pad_id=pad_id)
    test_ds  = IndexedBinaryDataset(args.test,  Config.seq_len, shuffle=False, repeat=False, pad_id=pad_id)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, num_workers=2, pin_memory=pin, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.batch_size, num_workers=1, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=Config.batch_size, num_workers=1, pin_memory=pin)

    # ---- Model ----
    model = GPT2LikeLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        dropout=Config.dropout,
        max_len=Config.max_len,
        pad_id=pad_id,
        bos_id=tok2id["<bos>"],
        eos_id=tok2id["<eos>"],
        unk_id=tok2id["<unk>"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(Config.adam_beta1, Config.adam_beta2),
        eps=Config.adam_eps,
        weight_decay=Config.weight_decay,
    )

    scheduler = get_lr_scheduler(
        optimizer,
        Config.warmup_steps,
        args.max_steps,
        min_lr=Config.epsilon,
    )
        
   # ---- Resume ----
    if args.resumePath and os.path.exists(args.resumePath):
        print(f"Resuming from checkpoint: {args.resumePath}")
        ckpt = torch.load(args.resumePath, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        if not args.override_hparams:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state"])
        else:
            print("⚡ Overriding hyperparameters: using new optimizer/scheduler")
            # re-init optimizer, scheduler with current Config
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(Config.adam_beta1, Config.adam_beta2),
                eps=Config.adam_eps,
                weight_decay=Config.weight_decay,
            )
            scheduler = get_lr_scheduler(
                optimizer,
                Config.warmup_steps,
                args.max_steps,
                min_lr=Config.epsilon,
            )
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step = ckpt.get("step", 0)
        print(f"✔️ Resumed at step {global_step} (best_val_loss={best_val_loss:.4f})")

    # --- Eval-only mode ---
    if args.evalPath and os.path.exists(args.evalPath) and args.evalSubset > 0:
        print(
            f"Evaluation mode. Evaluating {args.evalPath} with "
            f"{args.evalSubset * 100:.1f}% of eval set."
        )
        ckpt = torch.load(args.evalPath, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.eval()
        evaluate(
            model,
            test_loader,
            vocab_size,
            pad_id,
            device,
            "EVAL",
            max_batches=None,
            shard_frac=args.evalSubset,
        )
        sys.exit("Evaluation done")

    # ---- Training loop ----
    noImprovement = 0
    model.train()
    total_loss_sum, total_tokens = 0.0, 0

    while global_step < args.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            criterion = torch.nn.CrossEntropyLoss(
                ignore_index=pad_id,
                label_smoothing=Config.label_smoothing,
                reduction="mean",
            )
            loss = criterion(
                logits.view(-1, vocab_size),
                y.view(-1),
            ) / Config.gradAccumSteps
            
            loss.backward()

            if (global_step + 1) % Config.gradAccumSteps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            # Logging token-avg loss (accounting for gradAccum scaling)
            with torch.no_grad():
                valid = (y != pad_id).sum().item()
                # loss.item() is mean CE / gradAccumSteps; undo the division:
                mean_ce = loss.item() * Config.gradAccumSteps
                total_loss_sum += mean_ce * valid
                total_tokens += valid

            if Config.debug and global_step % Config.debug_every == 0:
                avg_tok_loss = total_loss_sum / total_tokens
                print(f"Step {global_step} - Train tokLoss={avg_tok_loss:.4f}")
                total_loss_sum, total_tokens = 0.0, 0

            # ---- Validation ----
            if global_step % args.eval_every_steps == 0:
                val_loss = evaluate(
                    model,
                    val_loader,
                    vocab_size,
                    pad_id,
                    device,
                    "VAL",
                    max_batches=Config.max_batches,
                    shard_frac=0.003,
                )
                if val_loss < best_val_loss - Config.improvement_threshold:
                    best_val_loss = val_loss
                    noImprovement = 0
                    save_model_state(model, optimizer, scheduler, global_step, "models/best_model.pt", "Best")
                else:
                    noImprovement += 1
                    print(f"No improvement. Patience {noImprovement}/{Config.patience}")
                    if noImprovement >= Config.patience:
                        print("⚠️ Early stopping")
                        break
                model.train()

            # ---- Save ----
            if global_step % args.save_every_steps == 0:
                os.makedirs("models", exist_ok=True)
                save_model_state(model, optimizer, scheduler, global_step, "models/last_save_state.pt", f"Step {global_step}")

            if global_step >= args.max_steps:
                break

    # ---- Final test evaluation ----
    test_loss = evaluate(
        model, test_loader, vocab_size, pad_id, device, "TEST", max_batches=None, shard_frac=1.0
    )
    print(f"✅ Training complete. Final Test Loss={test_loss:.4f}")


# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Train dataset prefix (shards)")
    parser.add_argument("--val",   type=str, required=True, help="Validation dataset prefix (shards)")
    parser.add_argument("--test",  type=str, required=True, help="Test dataset prefix (shards)")
    parser.add_argument("--vocab", type=str, required=True, help="Vocab JSON path")

    # defaults pulled from Config
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max_steps", type=int, default=Config.decay_steps)
    parser.add_argument("--eval_every_steps", type=int, default=Config.eval_every_steps)
    parser.add_argument("--save_every_steps", type=int, default=Config.save_every_steps)

    # Resume
    parser.add_argument("--resumePath", type=str, default="models/last_save_state.pt")
    parser.add_argument("--override_hparams", action="store_true",
        help="Ignore optimizer/scheduler state from checkpoint, use new Config hyperparams")
    
    # Eval only
    parser.add_argument("--evalPath", type=str, default="models/best_model.pt")
    parser.add_argument("--evalSubset", type=float, default=-0.1)
    
    args = parser.parse_args()

    main(args)