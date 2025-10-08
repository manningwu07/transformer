# server.py
import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformer import LLM
from params import Config
import json
from tokenizers import Tokenizer as HFTokenizer

# ---------- Config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "data/json/vocab.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")  # change if needed
TOK_PATH = os.path.join(BASE_DIR, "data/json/tokenizer.json")

# ---------- Load vocab ----------
print("CWD:", os.getcwd())
print("Looking for vocab:", os.path.abspath(VOCAB_PATH))

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
    
tok2id = vocab["TokenToID"]
id2tok = vocab["IDToToken"]
eos_id = tok2id["<eos>"]
hf_tok = HFTokenizer.from_file(TOK_PATH)

# ---------- Model ----------
vocab_size = len(id2tok)
model = LLM(
    vocab_size=vocab_size,
    d_model=Config.d_model,
    n_heads=Config.num_heads,
    n_layers=Config.n_layers,
    d_ff=Config.hidden_size,
    max_len=Config.max_len
)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state, strict=False)
model.eval()
model.eos_id = eos_id

model.load_state_dict(state, strict=False)
model.head.weight = model.tok_emb.weight

# ---------- FastAPI ----------
app = FastAPI()

# request schema
class InferRequest(BaseModel):
    ids: list[int]
    max_tokens: int = 20
    top_k: Optional[int] = 10
    top_p: Optional[float] = None
    temperature: float = 1.0
    repetition_penalty: float = 1.0  # default = no penalty


# helper: filtering logits
def filter_logits(logits, top_k=0, top_p=0.0):
    """Apply top-k and/or nucleus (top-p) filtering to logits"""
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # remove tokens with cumulative prob above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift right to keep first above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    return logits

@app.post("/generate")
def generate(req: InferRequest):
    try:
        ids = torch.tensor([req.ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
            ids,
            max_tokens=req.max_tokens,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
        )
        out_ids = out[0].tolist()

        # Robust byte-level decoding with the trained tokenizer
        decoded = hf_tok.decode(out_ids, skip_special_tokens=True)

        return {"ids": out_ids, "text": decoded.strip()}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
