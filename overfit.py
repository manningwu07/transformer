# overfit_wiki.py
import torch, json
from torch.utils.data import DataLoader
import numpy as np
from transformer import GPT2LikeLM
from params import Config

def collate_fn(batch):
    """Custom collate to pad sequences to same length"""
    max_len = max(len(x[0]) for x in batch)
    pad_id = batch[0][0].new_zeros(1).fill_(0).item()  # assume 0 is pad
    
    xs, ys = [], []
    for x, y in batch:
        if len(x) < max_len:
            pad_len = max_len - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=x.dtype)])
            y = torch.cat([y, torch.zeros(pad_len, dtype=y.dtype)])
        xs.append(x)
        ys.append(y)
    
    return torch.stack(xs), torch.stack(ys)

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, pad_id):
        idx = np.fromfile(f"{prefix}-000.idx",dtype=np.uint64).reshape(-1,2)
        data = np.memmap(f"{prefix}-000.bin",dtype=np.uint32,mode="r")
        self.samples=[]
        for start,length in idx:
            arr = data[start//4:start//4+length].astype(np.int64)
            x, y = arr[:-1], arr[1:]
            self.samples.append((x,y))
        self.pad_id=pad_id
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        x,y=self.samples[i]; return torch.tensor(x), torch.tensor(y)

def main():
    vocab=json.load(open("data/test/vocab.json"))
    pad_id=vocab["TokenToID"]["<pad>"]
    ds=PackedDataset("data/test/wiki_eval_ids",pad_id)
    loader=DataLoader(ds,batch_size=4,shuffle=True, collate_fn=collate_fn)

    model=GPT2LikeLM(
        vocab_size=len(vocab["IDToToken"]),
        d_model=Config.d_model,
        n_heads=Config.num_heads,
        n_layers=2,           # small for test
        d_ff=512,
        max_len=128,
        pad_id=pad_id,
        bos_id=vocab["TokenToID"]["<bos>"],
        eos_id=vocab["TokenToID"]["<eos>"],
        unk_id=vocab["TokenToID"]["<unk>"],
    )
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model=model.to(device)
    crit=torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    
    for step,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device)
        logits,_=model(x)
        loss=crit(logits.view(-1,logits.size(-1)),y.view(-1))
        loss.backward()
        if step % 50==0: print(f"step {step} loss={loss.item():.4f}")
        if step>500: break   # quick overfit
        
    # Try generate
    out=model.generate(torch.tensor([[vocab["TokenToID"]["<bos>"]]],device=device),max_tokens=40)
    ids=out[0].tolist()
    text="".join([vocab["IDToToken"][i] for i in ids if i in range(len(vocab["IDToToken"]))])
    print("GEN:",text)

if __name__=="__main__":
    main()