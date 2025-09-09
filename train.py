import argparse
import glob
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import signal
import sys

from transformer import GPT2LikeLM
from params import Config
import numpy as np


best_val_loss = float("inf")
model = None  # global ref for crash handler
optimizer = None
start_epoch = 1  # so we can resume


class IndexedBinaryDataset(torch.utils.data.IterableDataset):
    """
    Streams examples from .bin/.idx shards produced by ExportTokenIDsBinary.
    - .bin = int32 token IDs
    - .idx = (int64 start_offset, int64 length)
    """

    def __init__(self, prefix, seq_len, repeat=False, shuffle=False):
        self.prefix = prefix
        self.seq_len = seq_len
        self.repeat = repeat
        self.shuffle = shuffle

        # find shards
        self.shards = sorted(glob.glob(prefix + "-*.idx"))
        assert self.shards, f"No shards found for {prefix}"

    def _iter_shard(self, idx_path):
        bin_path = idx_path.replace(".idx", ".bin")
        idx_arr = np.fromfile(idx_path, dtype=np.int64).reshape(-1, 2)
        data = np.memmap(bin_path, dtype=np.int32, mode="r")

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


def load_vocab(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["TokenToID"], data["IDToToken"]


def save_model_state(model, optimizer, epoch, path, msg=""):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )
    print(f"💾 Saved {msg} → {path}")


def crash_handler(sig, frame):
    global model, optimizer, current_epoch
    if model is not None:
        os.makedirs("models", exist_ok=True)
        save_model_state(model, optimizer, current_epoch, "models/curr_model.pt", "CURRENT MODEL (crash-save)")
    sys.exit(0)


def main(args):
    global model, optimizer, best_val_loss, start_epoch, current_epoch

    # Handle Ctrl-C / SIGTERM to save model mid-run
    signal.signal(signal.SIGINT, crash_handler)
    signal.signal(signal.SIGTERM, crash_handler)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    vocab, id2tok = load_vocab(args.vocab)
    vocab_size = len(id2tok)
    print(f"Loaded vocab size: {vocab_size}")

    train_ds = IndexedBinaryDataset(args.train, Config.seq_len, shuffle=True, repeat=False)
    eval_ds = IndexedBinaryDataset(args.eval, Config.seq_len, shuffle=False, repeat=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        num_workers=2,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=Config.batch_size,
        num_workers=1,
    )

    # init model + optimizer
    model = GPT2LikeLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        max_len=Config.seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(Config.adam_beta1, Config.adam_beta2),
        eps=Config.adam_eps,
        weight_decay=Config.weight_decay,
    )

    # --- auto resume ---
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resumePath}")
        ckpt = torch.load(args.resumePath, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"✔️ Resumed from epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})")

    global_step = 0
    noImprovement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        current_epoch = epoch
        model.train()
        total_loss, total_tokens = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

            if Config.debug and global_step % Config.debug_every == 0:
                avg_tok_loss = total_loss / total_tokens
                ppl = math.exp(avg_tok_loss)
                print(f"Epoch {epoch}, Step {global_step} - Train tokLoss={avg_tok_loss:.4f}, PPL={ppl:.2f}")

        # --- validation ---
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item() * x.numel()
                val_tokens += x.numel()
        val_tok_loss = val_loss / val_tokens
        val_ppl = math.exp(val_tok_loss)
        
        end_time = time.time()  # Stop the timer
        epoch_duration = end_time - start_time
        
        print(f"Epoch {epoch} DONE → Val tokLoss={val_tok_loss:.4f}, PPL={val_ppl:.2f}, Time={epoch_duration:.2f}s")

        # -----Checking whether the model has improved-----
        if noImprovement >= Config.patience:
            print("⚠️ Early stopping triggered (no improvement within threshold)")
            break
        
        if val_tok_loss < best_val_loss - Config.improvement_threshold:
            # Significant improvement: update best loss, reset patience
            best_val_loss = val_tok_loss
            noImprovement = 0
            save_model_state("models/best_model.pt", epoch, "Best model")  # still save best
        else:
            # No significant improvement
            noImprovement += 1
            print(f"No significant improvement. Patience counter: {noImprovement}/{Config.patience}")

        # ---------------- saving ----------------
        os.makedirs("models", exist_ok=True)

        if epoch % Config.save_epoch_number == 0:
            save_model_state("models/last_save_state.pt", epoch, f"Epoch {epoch} (interval save)")
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required: dataset paths + vocab path
    parser.add_argument("--train", type=str, required=True, help="Path to training dataset (JSONL IDs)")
    parser.add_argument("--eval", type=str, required=True, help="Path to eval dataset (JSONL IDs)")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.json")

    # Optional: override Config defaults
    parser.add_argument("--epochs", type=int, default=Config.max_epochs, help=f"Number of epochs (default={Config.max_epochs})")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help=f"Batch size (default={Config.batch_size})")
    parser.add_argument("--lr", type=float, default=Config.attn_lr, help=f"Learning rate (default={Config.attn_lr})")
    parser.add_argument("--save", type=str, default="models/last_save_state.pt", help="Save path for checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resumePath", type=str, default="models/last_save_state.pt", help="Resume path for checkpoint")

    args = parser.parse_args()

    # Sync args back into Config so training loop picks up overrides
    Config.batch_size = args.batch_size
    Config.attn_lr = args.lr  # unify one LR for simplicity
    Config.max_epochs = args.epochs

    main(args)