#!/usr/bin/env python3
import argparse, json, os
import torch
from torch.utils.data import DataLoader
import numpy as np

from transformer_mlx import LLM
from params import Config
from train import IndexedBinaryDataset
from eval import evaluate

def main(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Training LoRA adapters on {device}")

    # --- Vocab ---
    vocab = json.load(open(args.vocab))
    tok2id, id2tok = vocab["TokenToID"], vocab["IDToToken"]
    vocab_size = len(id2tok)
    pad_id = tok2id["<pad>"]

    # --- Dataset ---
    train_ds = IndexedBinaryDataset(args.train, Config.seq_len, shuffle=True, pad_id=pad_id)
    val_ds = IndexedBinaryDataset(args.val, Config.seq_len, shuffle=False, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, num_workers=1)
    val_loader   = DataLoader(val_ds,   batch_size=Config.batch_size, num_workers=1)

    # --- Model ---
    model = LLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        dropout=Config.dropout,
        max_len=Config.max_len
    ).to(device)
    
    model.load_state_dict(state, strict=False)
    model.head.weight = model.tok_emb.weight

    # --- Load base weights ---
    ckpt = torch.load(args.resumePath, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    print("✔️ Loaded base checkpoint.")

    # Freeze everything except LoRA params
    for name, p in model.named_parameters():
        if "A" in name or "B" in name:  # LoRA adapters
            p.requires_grad = True
        else:
            p.requires_grad = False

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    # --- Training ---
    steps = 0
    model.train()
    while steps < args.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 100 == 0:
                print(f"Step {steps} LoRA loss={loss.item():.4f}")

            if steps % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, vocab_size, pad_id, device, split="VAL", max_batches=100)
                print(f"[VAL-LORA] step {steps}, val_loss={val_loss:.4f}")

            if steps >= args.max_steps:
                break

    # Save only LoRA adapter weights
    lora_state = {k: v for k, v in model.state_dict().items() if "A" in k or "B" in k}
    os.makedirs("models", exist_ok=True)
    torch.save(lora_state, "models/lora_alpaca.pt")
    print("💾 Saved LoRA adapters to models/lora_alpaca.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--resumePath", type=str, default="models/best_model.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()
    main(args)