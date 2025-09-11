import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # query,key,value
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Register a causal mask (upper triangular)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len), persistent=False)

    def forward(self, x, past_kv=None):
        # x: (B, T, d_model)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B,T,3*d)
        q, k, v = qkv.split(C, dim=2)

        # reshape into heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,T,d)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # If we have past KV (for incremental decoding), append
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)  # (B,h,T_total,d)
            v = torch.cat([pv, v], dim=2)

        # save for cache
        present = (k, v)

        # attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,h,T,T_total)
        # apply causal mask
        T_q = q.size(2)
        T_k = k.size(2)
        mask = self.mask[:, :, :T_q, :T_k]  # (1,1,T_q,T_k)
        # use a large negative for numerical stability on MPS
        att = att.masked_fill(~mask, -1e9)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # back to (B,T,C)
        y = self.resid_drop(self.out(y))  # linear out proj

        return y, present


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln1(x), past_kv)
        x = x + a
        m = self.mlp(self.ln2(x))
        x = x + m
        return x, present


class GPT2LikeLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=1024,
        dropout=0.1,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, max_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        
        
        # store special token IDs
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def forward(self, idx, past_kvs=None):
        B, T = idx.size()
        past_len = 0
        if past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None:
            # past_kvs[0][0] is k: (B,h,T_past,d)
            past_len = past_kvs[0][0].size(2)
        pos = torch.arange(past_len, past_len + T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, present = block(x, past)
            new_kvs.append(present)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_tokens=50, top_k=None, temperature=1.0):
        self.eval()
        past_kvs = None
        for _ in range(max_tokens):
            logits, past_kvs = self(idx[:, -1:], past_kvs)
            logits = logits[:, -1, :] / temperature

            # Mask unwanted tokens
            for tok in [self.pad_id, self.unk_id, self.bos_id]:
                logits[:, tok] = -1e9

            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -1e9

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if next_token.item() == self.eos_id:
                break
        return idx