# inspect_logits.py
import argparse, torch, json, numpy as np
from tokenizers import Tokenizer
from transformer_mlx import LLM

def stats(t):
    return dict(min=float(t.min()), max=float(t.max()), mean=float(t.mean()), std=float(t.std()), nan=bool(torch.isnan(t).any()), inf=bool(torch.isinf(t).any()))

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tok", required=True)
parser.add_argument("--vocab", default="data/json/vocab.json")
parser.add_argument("--prompt", default="Hello, how are you?")
args = parser.parse_args()

tok = Tokenizer.from_file(args.tok)
vocab = json.load(open(args.vocab))
vocab_size = len(vocab["IDToToken"])

print("Loading model...")
ckpt = torch.load(args.ckpt, map_location="cpu")
state = ckpt.get("model_state", ckpt)
# Build model with your Config; adjust if your params module is named differently
from params import Config
model = LLM(vocab_size=vocab_size,
                   d_model=Config.d_model,
                   n_heads=Config.num_heads,
                   n_layers=Config.n_layers,
                   d_ff=Config.hidden_size,
                   max_len=Config.max_len)
model.load_state_dict(state, strict=False)
model.eval()

model.load_state_dict(state, strict=False)
model.head.weight = model.tok_emb.weight
    

# tokenize
enc = tok.encode(args.prompt)
ids = [tok.token_to_id("<bos>")] + enc.ids
ids_t = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    logits, _ = model(ids_t)          # logits: (B, T, V)
    last_logits = logits[:, -1, :]    # examine last-step logits

print("Logits shape:", tuple(logits.shape))
print("Logits stats (full):", stats(logits))
print("Last-step logits stats:", stats(last_logits))

# softmax sums and NaNs
probs = torch.softmax(last_logits, dim=-1)
print("Softmax sums (per-batch item, last step):", probs.sum(dim=-1).tolist())
print("Softmax stats (last step):", stats(probs))

# topk diagnostics
topk = torch.topk(probs, k=20)
for i,v in enumerate(topk.values[0].tolist()):
    print(f"top{i+1} prob = {v:.6f}")
cum_top10 = topk.values[0,:10].sum().item()
print("Cumulative top-10 prob:", cum_top10)

# entropy
ent = - (probs * (probs + 1e-30).log()).sum(-1)
print("Entropy (nats) of last-step distribution:", ent.tolist())

# check head weight shape
print("head weight shape:", tuple(model.head.weight.shape))
print("vocab_size (from vocab file):", vocab_size)