import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from transformer import LLM
from params import Config
from torch.utils.data import DataLoader
# Assuming you use the IndexedBinaryDataset from your train.py logic
from train import IndexedBinaryDataset 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            # Note: transformer.py returns (logits, loss) if targets provided
            _, loss = model(x, targets=y)
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
            # Update progress bar with current perplexity
            current_ppl = torch.exp(torch.tensor(total_loss / total_tokens))
            pbar.set_postfix({"PPL": f"{current_ppl:.2f}"})

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--data", type=str, default="data/shards/val", help="Path to validation shards")
    args = parser.parse_args()

    print(f"📉 Evaluating Checkpoint: {args.ckpt}")
    
    # 1. Load Model
    model = LLM()
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    else:
        print("❌ Checkpoint not found.")
        return
        
    model.to(DEVICE).to(dtype=torch.bfloat16) # BF16 for speed/memory

    # 2. Load Data
    # Ensure validation data exists
    if not os.path.exists(args.data):
        print(f"❌ Validation data not found at {args.data}")
        return

    # Use a larger batch size for eval since no gradients are stored
    val_ds = IndexedBinaryDataset(args.data, Config.seq_len)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size * 2, num_workers=4)
    
    # 3. Run Eval
    loss, ppl = evaluate(model, val_loader)
    
    print("-" * 40)
    print(f"✅ Final Results:")
    print(f"   Validation Loss: {loss:.4f}")
    print(f"   Perplexity:      {ppl:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()