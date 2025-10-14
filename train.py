#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
import signal
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from eval import evaluate
from transformer import LLM
from params import Config
# from transformers import Adafactor
# from transformers.optimization import get_constant_schedule_with_warmup
from torch.optim import AdamW


# --------------------
# Dataset utilities
# --------------------

class IndexedBinaryDataset(torch.utils.data.IterableDataset):
    def __init__(self, prefix, seq_len, repeat=True, shuffle=True, pad_id=0, seed=42):
        self.prefix = prefix
        self.seq_len = seq_len
        self.repeat = repeat
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.shards = sorted(glob.glob(prefix + "-*.idx"))
        assert self.shards, f"No shards found for {prefix}"
        self.rng = random.Random(seed)
        self._used_shards = set()

    def _select_next_shard(self):
        all_idx = list(range(len(self.shards)))
        available = [i for i in all_idx if i not in self._used_shards]
        if not available:
            # all shards consumed → reset epoch
            self._used_shards.clear()
            available = all_idx
        shard_id = self.rng.choice(available)
        self._used_shards.add(shard_id)
        return self.shards[shard_id]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1

        while True:
            idx_path = self._select_next_shard()
            bin_path = idx_path.replace(".idx", ".bin")
            idx_arr = np.fromfile(idx_path, dtype=np.uint64).reshape(-1, 2)
            data = np.memmap(bin_path, dtype=np.uint32, mode="r")

            # optional random starting position within shard
            offset = 0
            if self.shuffle:
                offset = self.rng.randint(0, len(idx_arr) - 1)

            order = list(np.arange(len(idx_arr)))
            if self.shuffle:
                self.rng.shuffle(order)
            order = order[offset:] + order[:offset]

            # split order among workers
            chunk = len(order) // nworkers
            start = wid * chunk
            end = len(order) if wid == nworkers - 1 else (wid + 1) * chunk
            for j in order[start:end]:
                st, ln = idx_arr[j]
                arr = data[st // 4 : st // 4 + ln]
                ids = torch.from_numpy(arr.astype(np.int64))
                if len(ids) < 2:
                    continue
                if len(ids) > self.seq_len:
                    if self.shuffle:
                        start_off = self.rng.randint(0, len(ids) - self.seq_len)
                    else:
                        start_off = 0
                    ids = ids[start_off : start_off + self.seq_len]
                elif len(ids) < self.seq_len:
                    pad = torch.full((self.seq_len - len(ids),), self.pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad])
                yield ids[:-1], ids[1:]

            if not self.repeat and len(self._used_shards) == len(self.shards):
                break

# --------------------
# Vocab utils
# --------------------

def load_vocab(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["TokenToID"], data["IDToToken"]


# -------------------------------
# Custom cosine scheduler (manual update, no LambdaLR)
# -------------------------------
def update_lr(optimizer, step, curr_tokens_seen=0):
    """
    Adjusts LR using:
      - linear warm-up until Config.target_warmup_tokens tokens processed
      - cosine decay thereafter based on optimizer step progression
    """
    start = Config.startLr
    end = Config.endLr
    total_steps = max(1, Config.totalOptSteps)

    # --- Warm-up phase (token-based) ---
    if curr_tokens_seen < Config.target_warmup_tokens:
        progress = curr_tokens_seen / max(1.0, Config.target_warmup_tokens)
        new_lr = start * progress
    else:
        # --- Cosine decay after warm-up ---
        progress = min(step / total_steps, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        new_lr = end + (start - end) * cosine

    # Update optimizer groups
    for g in optimizer.param_groups:
        g["lr"] = new_lr

    return new_lr


# --------------------
# Checkpoint helpers
# --------------------

def save_model_state(model, optimizer, step, path, tokens_seen, msg=""):
    print(f"Model save checkpoint: opt_step={opt_step}, tokens_seen={tokens_seen}")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "tokens_seen_total": tokens_seen
        },
        path,
    )
    print(f"💾 Saved {msg} → {path}")


def crash_handler(sig, frame):
    global model, optimizer, opt_step, tokens_seen
    print(f"Model crash checkpoint: opt_step={opt_step}, tokens_seen={tokens_seen}")
    if model is not None:
        os.makedirs("models", exist_ok=True)
        save_model_state(
            model,
            optimizer,
            opt_step,
            "models/curr_model.pt",
            tokens_seen,
            "CURRENT MODEL (crash-save)",
        )
    sys.exit(0)



# --------------------
# Training loop
# --------------------

best_val_loss = float("inf")
model = None
optimizer = None
opt_step = 1
tokens_seen = 0


def main(args):
    global model, optimizer, best_val_loss, opt_step, tokens_seen
    torch.set_num_threads(os.cpu_count() - 2 if os.cpu_count() > 2 else 0)

    # Handle Ctrl-C / SIGTERM
    signal.signal(signal.SIGINT, crash_handler)
    signal.signal(signal.SIGTERM, crash_handler)

    device = "mps"
    is_mps = device == "mps"
    
    print(f"Training on: {device}")
    
    # ---- Load prebuilt bad token IDs (for generation masking) ----
    BAD_PATH = os.path.join("data", "json", "bad_ids.json")
    if os.path.exists(BAD_PATH):
        with open(BAD_PATH, "r") as f:
            bad_json = json.load(f)
        BAD_IDS = bad_json.get("BadTokenIDs", [])
        print(f"⚙️ Loaded {len(BAD_IDS)} bad token IDs from {BAD_PATH}")
    else:
        BAD_IDS = []
        print(f"⚠️  bad_ids.json not found at {BAD_PATH} (generation mask will be empty)")

    # Remove any accidental duplicates (keep ordering stable)
    BAD_IDS = list(dict.fromkeys(BAD_IDS))

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
    # On macOS/MPS, multiprocessing often hurts more than helps for memmap IO.
    nw_train = 0 if is_mps else 2
    nw_eval = 0 if is_mps else 1
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        num_workers=nw_train,
        pin_memory=pin,
        persistent_workers=False if is_mps else True,
    )
    val_loader   = DataLoader(val_ds,   batch_size=Config.batch_size, num_workers=nw_eval, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=Config.batch_size, num_workers=nw_eval, pin_memory=pin)

    # ---- Model ----
    model = LLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        dropout=Config.dropout,
        max_len=Config.max_len
    ).to(device)
    
    # Proper weight tying: use the SAME Parameter object (no duplicate grads)
    model.head.weight = model.tok_emb.weight
    assert model.head.weight is model.tok_emb.weight
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    print("✅ weight tied (same Parameter object)")

    # Compile helps on CUDA; on MPS it usually hurts (graph breaks, fallbacks).
    if device == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            print("✅ torch.compile enabled")
        except Exception as e:
            print("⚠️ torch.compile not available:", e)
    else:
        print("ℹ️ Skipping torch.compile on non-CUDA device")
    
    # Use AdamW with a fixed LR for now because Adafactor isnt worth it at a smaller model. 
    optimizer = AdamW(
        model.parameters(),
        lr=Config.startLr,
        betas=(0.9, 0.95),
        weight_decay=Config.weight_decay,
        eps=1e-8,
    )
        
    # ---- Resume ----
    if args.resumePath and os.path.exists(args.resumePath):
        print(f"Resuming from checkpoint: {args.resumePath}")
        ckpt = torch.load(args.resumePath, map_location=device)
        state = ckpt.get("model_state", ckpt)
        
        model.load_state_dict(state, strict=False)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        tokens_seen = 0 # ckpt.get("tokens_seen_total", 0)
        opt_step = int(ckpt.get("step", 1))
        try:
            model.head.weight = model.tok_emb.weight
            assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
        except Exception as e:
            print("⚠️  Weight-tying check failed:", e)
        
        if not args.override_hparams:
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as _:
                    print("ℹ️ Optimizer type or shapes differ; starting fresh optimizer")
        else:
            print("⚡ Overriding hyperparameters: using new optimizer")
            optimizer = AdamW(
                model.parameters(),
                lr= (Config.startLr - Config.endLr) / 2, # Manually change this as backup LR if update_lr doesnt work properly
                betas=(Config.beta1, Config.beta2),
                weight_decay=Config.weight_decay,
                eps=1e-8,
            )
            update_lr(optimizer, opt_step, tokens_seen)        
            
        # Logging statements to make sure everythings okay
        print(
            f"✔️ Resumed training from checkpoint."
            f" opt_step≈{ckpt.get('step',0)}, tokens_seen_total={tokens_seen:.3e},"
            f" best_val_loss={best_val_loss:.4f}"
            f" curr_adamW_lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        print(f"✔️ (best_val_loss={best_val_loss:.4f})")

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
    steps_done = 0
    t0 = time.time()
    tokens_seen_curr_run = 0
    model.train()
    total_loss_sum, total_tokens = 0.0, 0
    
    # 🧮 AMP GradScaler for mixed precision (bf16 on MPS, fp16 on CUDA)
    amp_dtype = torch.bfloat16 if device == "mps" else torch.float16
    
    # Ensure weight tied after scaler init (in case it changes storage)
    print("head ptr :", model.head.weight.data_ptr())
    print("tok_emb ptr:", model.tok_emb.weight.data_ptr())
    print("shared   :", model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr())
    
    tokens_per_opt = Config.batch_size * Config.seq_len * Config.gradAccumSteps
    print(f"Tokens per optimizer step: {tokens_per_opt:,}")

    # Compute equivalent warm‑up steps just for info
    warmup_equiv = int(Config.target_warmup_tokens / tokens_per_opt)
    print(f"≈ {warmup_equiv:,} optimizer steps ≈ {Config.target_warmup_tokens:.2e} target warm-up tokens")

    stop_training = False
    while opt_step < args.max_steps and not stop_training:
        for step_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # Sanity check
            # if step_idx % 1 == 0 and opt_step < 3:
            #     print(f"Step {step_idx}: x.device={x.device}, model.device={next(model.parameters()).device}, opt_step={opt_step}")
    
            x, y = x.to(device), y.to(device)
            criterion = torch.nn.CrossEntropyLoss(
                ignore_index=pad_id,
                label_smoothing=Config.label_smoothing,
                reduction="mean",
            )
            
            # Autocast only on CUDA; MPS autocast support is limited and can be slower.
            if device == "cuda":
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()

            with ctx:
                logits, _ = model(x)
                # True token-level CE (use this for debug/logging only)
                ce_loss = criterion(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = ce_loss / Config.gradAccumSteps

            loss.backward()
            n_nonpad = (y != pad_id).sum().item()
            total_loss_sum += ce_loss.detach().item() * max(1, n_nonpad)
            total_tokens   += n_nonpad
            
            if (step_idx + 1) % Config.gradAccumSteps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
                optimizer.step()
                curr_lr = update_lr(optimizer, opt_step, tokens_seen)
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                tokens_seen += tokens_per_opt
                tokens_seen_curr_run += tokens_per_opt
                steps_done += 1
            
                # ---- Logging (once per optimizer step) ----
                if Config.debug and opt_step % Config.debug_every == 0 and opt_step > 0:
                    avg_tok_loss = total_loss_sum / max(1, total_tokens)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[{now}] [DEBUG] opt_step={opt_step:<6d}  "
                        f"tokens_seen={tokens_seen:.3e}  "
                        f"tokLoss={avg_tok_loss:.4f}  "
                        f"adamW_lr={curr_lr:.6f} "
                    )
                    total_loss_sum, total_tokens = 0.0, 0.0
                    # Throughput
                    dt = time.time() - t0
                    sps = steps_done / max(1e-6, dt)
                    toks_s = tokens_seen_curr_run / max(1e-6, dt)
                    print(f"⏱ {sps:.2f} opt_steps/s, {toks_s:.0f} toks/s (running)")
                    
                # ---- Validation (only right after an optimizer step) ----
                if (opt_step > 0) and (opt_step % args.eval_every_steps == 0):
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
                        save_model_state(model, optimizer, opt_step, "models/best_model.pt", tokens_seen, "Best")
                    else:
                        noImprovement += 1
                        print(f"No improvement. Patience {noImprovement}/{Config.patience}")
                        if noImprovement >= Config.patience:
                            print("⚠️ Early stopping")
                            stop_training = True
                            save_model_state(
                                model,
                                optimizer,
                                opt_step,
                                "models/early_stopping.pt",
                                tokens_seen,
                                f"opt_step {opt_step}",
                            )
                            break
                    model.train()
        # break outer while if early stopped
        if stop_training:
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
    parser.add_argument("--max_steps", type=int, default=Config.totalOptSteps)
    parser.add_argument("--eval_every_steps", type=int, default=Config.eval_every_steps)

    # Resume
    parser.add_argument("--resumePath", type=str, default="models/last_save_state.pt")
    parser.add_argument("--override_hparams", action="store_true",
        help="Ignore optimizer/scheduler state from checkpoint, use new Config hyperparams")
    
    # Eval only
    parser.add_argument("--evalPath", type=str, default="models/best_model.pt")
    parser.add_argument("--evalSubset", type=float, default=-0.1)
    
    args = parser.parse_args()

    main(args)