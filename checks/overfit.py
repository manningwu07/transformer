# overfit_wiki.py
import os
from tokenizers import Tokenizer
import torch, json
from torch.utils.data import DataLoader
import numpy as np
from transformer_mlx import LLM
from params import Config

def collate_fn(batch, pad_id):
    # tokenizer = Tokenizer.from_file("data/test/tokenizer.json")
    max_len = max(len(x[0]) for x in batch)
    xs, ys = [], []
    for x, y in batch:
        if len(x) < max_len:
            pad_len = max_len - len(x)
            x = torch.cat([x, torch.full((pad_len,), pad_id, dtype=x.dtype)])
            y = torch.cat([y, torch.full((pad_len,), pad_id, dtype=y.dtype)])
        xs.append(x)
        ys.append(y)
    
    return torch.stack(xs), torch.stack(ys)

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, pad_id, max_samples=3):
        idx = np.fromfile(f"{prefix}-000.idx", dtype=np.uint64).reshape(-1,2)
        data = np.memmap(f"{prefix}-000.bin", dtype=np.uint32, mode="r")
        self.samples = []
        for i, (start,length) in enumerate(idx[:max_samples]):
            arr = data[start//4:start//4+length].astype(np.int64)
            x, y = arr[:-1], arr[1:]
            self.samples.append((x,y))
        self.pad_id = pad_id
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        x,y=self.samples[i]; return torch.tensor(x), torch.tensor(y)

def main():    
    vocab = json.load(open("data/json/vocab.json"))
    pad_id = vocab["TokenToID"]["<pad>"]

    # small sample from your new shards for smoke test (choose any split)
    ds = PackedDataset("data/shards/val", pad_id, max_samples=8)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b,pad_id))

    model=LLM(
        vocab_size=len(vocab["IDToToken"]),
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=2,           # small for test
        d_ff=Config.hidden_size,
        max_len=1024
    )
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model=model.to(device)
    crit=torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(10):
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device)
            logits,_=model(x)
            loss=crit(logits.view(-1,logits.size(-1)),y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 ==0: print(f"step {step} loss={loss.item():.4f}")
            if step>1000: break   # quick overfit
        
    # Try generate
    tok = Tokenizer.from_file("data/json/tokenizer.json")

    bad_path = "data/json/bad_ids.json"
    bad_ids = []
    if os.path.exists(bad_path):
        bad_ids = json.load(open(bad_path)).get("BadTokenIDs", [])

    out = model.generate(
        torch.tensor([[vocab["TokenToID"]["<bos>"]]], device=device),
        max_tokens=40,
        top_k=40,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
    )
    ids = [i for i in out[0].tolist() if i not in bad_ids]
    text = tok.decode(ids, skip_special_tokens=True)
    print("GEN:", text)
    print("GEN:",text)

if __name__=="__main__":
    main()