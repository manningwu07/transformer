# server.py
import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformer import LLM
from params import Config
from tokenizers import Tokenizer as HFTokenizer

# ---------- Config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_step_*.pt")  # or specific checkpoint
TOK_PATH = os.path.join(BASE_DIR, "data/json/tokenizer.json")

# ---------- Device ----------
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------- Load Model ----------
print(f"⏳ Loading model on {DEVICE}...")
model = LLM(Config)

# Find latest checkpoint
import glob
ckpt_files = sorted(glob.glob(MODEL_PATH), key=os.path.getmtime)
if ckpt_files:
    ckpt = torch.load(ckpt_files[-1], map_location="cpu")
    # Handle both formats: {"model": state_dict} or raw state_dict
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"✅ Loaded: {ckpt_files[-1]}")
else:
    print("⚠️ No checkpoint found, using random weights")

model.to(DEVICE).to(dtype=torch.bfloat16)
model.eval()

# ---------- Load Tokenizer ----------
hf_tok = HFTokenizer.from_file(TOK_PATH)
vocab_size = hf_tok.get_vocab_size()

# ---------- FastAPI ----------
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: Optional[int] = 50

class TokenRequest(BaseModel):
    ids: list[int]
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: Optional[int] = 50

@app.post("/generate")
def generate(req: GenerateRequest):
    # Encode prompt
    encoded = hf_tok.encode(req.prompt)
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=DEVICE)
    
    # Generate
    generated = []
    with torch.no_grad():
        for next_token in model.generate(
            ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            use_cache=True
        ):
            tok_id = next_token.item()
            generated.append(tok_id)
            # Stop on EOS
            if tok_id == hf_tok.token_to_id("<eos>"):
                break
    
    # Decode
    full_ids = encoded.ids + generated
    text = hf_tok.decode(full_ids, skip_special_tokens=True)
    
    return {"text": text.strip(), "ids": full_ids}

@app.post("/generate_from_ids")
def generate_from_ids(req: TokenRequest):
    ids = torch.tensor([req.ids], dtype=torch.long, device=DEVICE)
    
    generated = []
    with torch.no_grad():
        for next_token in model.generate(
            ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            use_cache=True
        ):
            generated.append(next_token.item())
    
    full_ids = req.ids + generated
    text = hf_tok.decode(full_ids, skip_special_tokens=True)
    
    return {"text": text.strip(), "ids": full_ids}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "vocab_size": vocab_size}

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000