# inspect_logits_raw.py
import torch, json
from tokenizers import Tokenizer
from transformer import GPT2LikeLM
from params import Config

ckpt_path = "models/best_model.pt"
tok_path = "data/json/tokenizer.json"
vocab_path = "data/json/vocab.json"
prompt = "Hello, how are you?"

tok = Tokenizer.from_file(tok_path)
vocab = json.load(open(vocab_path))
vocab_size = len(vocab["IDToToken"])

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt.get("model_state", ckpt)

model = GPT2LikeLM(
    vocab_size=vocab_size,
    d_model=Config.d_model,
    n_heads=Config.num_heads,
    n_layers=Config.n_layers,
    d_ff=Config.hidden_size,
    max_len=Config.max_len,
)
model.load_state_dict(state, strict=False)
model.eval()

enc = tok.encode(prompt)
ids = [tok.token_to_id("<bos>")] + enc.ids
ids_t = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    logits, _ = model(ids_t)
    last = logits[:, -1, :]    # shape (1, V)
    last = last.squeeze(0)

# Basic stats
print("logits min/max/mean/std:", float(last.min()), float(last.max()), float(last.mean()), float(last.std()))
# Topk raw logits and indices
topk = torch.topk(last, k=10)
for i,(val,idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
    print(f"top{i+1}: idx={idx} logit={val} tok={vocab['IDToToken'][idx] if idx < len(vocab['IDToToken']) else 'UNK'}")
# gaps
if topk.values.size(0) >= 2:
    print("gap top1 - top2:", float(topk.values[0]-topk.values[1]))
    
we = model.tok_emb.weight
hw = model.head.weight
print("embeddings norm:", float(we.norm().item()))
print("head weight norm:", float(hw.norm().item()))

bos_id = tok.token_to_id("<bos>")
print("<bos> id:", bos_id)
print("tok_emb[bos][:8]:", we[bos_id][:8].tolist())
print("head weight row for argmax idx printed above available if you share it.")