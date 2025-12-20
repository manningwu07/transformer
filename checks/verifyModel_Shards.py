import argparse
import numpy as np
import json
import torch
from transformer_mlx import LLM
from params import Config


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

# Show Model shape 
def verifyModel():
    with open(args.vocab, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["IDToToken"])
    print(f"🔤 Vocab size: {vocab_size}")

    # Load checkpoint
    ckpt = torch.load(args.model, map_location="cpu")
    model_state = ckpt.get("model_state", ckpt)  # support raw state_dict

    # Build model with config
    model = LLM(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=Config.n_layers,
        d_ff=Config.hidden_size,
        max_len=Config.max_len
    )
    

    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Print model structure summary
    print("\n📐 Model Architecture")
    print(model)

    # Shapes of critical parameters
    print("\n🔎 Key weight shapes:")
    print("tok_emb:", tuple(model.tok_emb.weight.shape))
    print("pos_emb:", tuple(model.pos_emb.weight.shape))
    print("head:", tuple(model.head.weight.shape))

    # Parameter counts
    total_params, trainable_params = count_params(model)
    print(f"\n🧮 Params: {total_params:,} total ({trainable_params:,} trainable)")

    # Layers / heads info
    print("\n⚙️ Config:")
    print(f"d_model = {Config.d_model}")
    print(f"hidden_size (MLP) = {Config.hidden_size}")
    print(f"num_heads = {Config.num_heads}")
    print(f"n_layers = {Config.n_layers}")
    print(f"seq_len = {Config.seq_len}")
    
    
# Make sure data is correct shape and contents are not corrupted
def verifyShards(shardNumber):
    a = np.memmap(f"data/shards/train-00{shardNumber}.bin", dtype=np.int32, mode="r")
    ids, counts = np.unique(a[:500000], return_counts=True)
    print("Unique IDs:", len(ids))
    print("Most common:", sorted(zip(counts, ids), reverse=True)[:30])


def main():
    try:
        if args.shardNumber >= 0:
            verifyShards(args.shardNumber)
        elif args.model and args.vocab:
            verifyModel()
        else:
            print("Missing --shardNumber or --model and --vocab argument(s)")
    except:
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shardNumber", type=int, default=-1)
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Path to .pt checkpoint file")
    parser.add_argument("--vocab", type=str, default="data/json/vocab.json", help="Vocab JSON path")
    args = parser.parse_args()