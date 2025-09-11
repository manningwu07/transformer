# server.py
import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformer import GPT2LikeLM 
from params import Config
import json

# ---------- Config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "data/test/vocab.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "curr_model.pt") #change this if nessecary

# ---------- Load vocab ----------
print("CWD:", os.getcwd())
print("Looking for vocab:", os.path.abspath(VOCAB_PATH))

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
tok2id = vocab["TokenToID"]
id2tok = vocab["IDToToken"]
eos_id = tok2id["<eos>"]

# ---------- Model ----------
vocab_size = len(id2tok)
model = GPT2LikeLM(
    vocab_size=vocab_size,
    d_model=Config.d_model,
    n_heads=Config.num_heads,
    n_layers=Config.n_layers,
    d_ff=Config.hidden_size,
    max_len=Config.seq_len,
    pad_id=tok2id["<pad>"],
    bos_id=tok2id["<bos>"],
    eos_id=tok2id["<eos>"],
    unk_id=tok2id["<unk>"],
)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state, strict=False)
model.eval()
model.eos_id = eos_id

# ---------- FastAPI ----------
app = FastAPI()

# request schema
class InferRequest(BaseModel):
    ids: list[int]
    max_tokens: int = 20
    top_k: Optional[int] = 10
    temperature: float = 1.0


@app.post("/generate")
def generate(req: InferRequest):
    try:
        ids = torch.tensor([req.ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_tokens=req.max_tokens,
                top_k=req.top_k,
                temperature=req.temperature,
            )
        out_ids = out[0].tolist()
        
        tokens = [id2tok[i] for i in out_ids]

        specials = {"<pad>", "<bos>", "<unk>"}
        decoded = "".join(tok if tok not in specials else "" for tok in tokens)
        
        return {"ids": out_ids, "text": decoded.strip()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}