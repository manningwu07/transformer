import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import LoRALinear
from params import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=2048, return_present=False, lora_r=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        qkv = nn.Linear(d_model, 3 * d_model)
        if lora_r:
            # split into q, k, v projections
            q_layer = nn.Linear(d_model, d_model)
            k_layer = nn.Linear(d_model, d_model)
            v_layer = nn.Linear(d_model, d_model)
            # LoRA only on q and v
            self.q_proj = LoRALinear(q_layer, r=lora_r)
            self.k_proj = k_layer
            self.v_proj = LoRALinear(v_layer, r=lora_r)
            self.merge = nn.Linear(3 * d_model, 3 * d_model)
        else:
            self.qkv = qkv

        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.return_present = return_present

        # Register a causal mask (upper triangular)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len), persistent=False)

    def forward(self, x, past_kv=None):
        # x: (B, T, d_model)
        B, T, C = x.size()
        if hasattr(self, "qkv"):  # normal mode
            qkv = self.qkv(x)  # (B,T,3*d)
            q, k, v = qkv.split(C, dim=2)
        else:  # LoRA mode
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

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
        
        B, _, T_q, _ = q.shape
        T_k = k.size(2)
        if T_k > self.mask.size(-1):
            device = self.mask.device
            self.mask = torch.tril(torch.ones(T_k, T_k, dtype=torch.bool, device=device)).view(1,1,T_k,T_k)

        if (Config.seq_len <= 256):
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            mask = self.mask[:, :, :T_q, :T_k]
            att = att.masked_fill(~mask, -1e9)
            att = F.softmax(att, dim=-1)
            att = torch.nan_to_num(att, nan=0.0)
            att = self.attn_drop(att)
            y = att @ v  # (B,h,T,d) 
        else:
            # Use SDPA's built-in causal mask. Do not pass a boolean "allow" mask.
            # This avoids the True=masked vs True=allowed semantics mismatch.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError("NaN/Inf in attention output")

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # back to (B,T,C)
        y = self.resid_drop(self.out(y))  # linear out proj

        if self.return_present:
            return y, present
        else:
            return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_len, return_present=False):
        super().__init__()
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(dim))
                self.eps = eps
            def forward(self, x):
                norm = x.pow(2).mean(-1, keepdim=True)
                return self.scale * x / torch.sqrt(norm + self.eps)

        # --- SwiGLU MLP: efficient and higher‑quality
        class SwiGLU(nn.Module):
            def __init__(self, d_model, d_ff, dropout):
                super().__init__()
                self.fc1 = nn.Linear(d_model, 2 * d_ff)
                self.fc2 = nn.Linear(d_ff, d_model)
                self.drop = nn.Dropout(dropout)
            def forward(self, x):
                x_gated = self.fc1(x)
                x_g, x_v = x_gated.chunk(2, dim=-1)
                return self.drop(self.fc2(F.silu(x_g) * x_v))

        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)
        
        self.return_present = return_present

    def forward(self, x, past_kv=None):
        if self.return_present == False:
            # checkpoint saves memory – only need y ("a")
            a = torch.utils.checkpoint.checkpoint(
                lambda t: self.attn(self.ln1(t), past_kv),
                x,
                use_reentrant=False,
            )
            x = x + a
            m = torch.utils.checkpoint.checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
            x = x + m
            return x, None
        else:
            # inference path
            a, present = self.attn(self.ln1(x), past_kv, return_present=True)
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
        dropout=0.1
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, max_len, return_present=False) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        
        self.head.weight = self.tok_emb.weight  # weight tying
    
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
    def generate(
        self,
        idx,
        max_tokens=20,
        top_k=15,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.3,
    ):
        """
        Incremental decoding with KV cache.
        - First step uses the full prompt to build cache.
        - Subsequent steps feed only the last token.
        - Supports top-k, top-p, repetition penalty, temperature.
        """
        self.eval()
        device = next(self.parameters()).device
        generated = idx.to(device)
        past_kvs = None
        
        bad_ids = (lambda p: json.load(open(p)).get("BadTokenIDs", []) if os.path.exists(p) else [])("data/test/bad_ids.json")
        if not bad_ids:
            FileNotFoundError("Bad IDs file not found or isn't correct. Please run the tokenizer script to generate bad_ids.json")

        def filter_top_k_p(logits, top_k, top_p):
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(
                    logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits
                )
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                to_remove = cum > top_p
                to_remove[:, 1:] = to_remove[:, :-1].clone()
                to_remove[:, 0] = 0
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, sorted_idx, to_remove)
                logits = logits.masked_fill(mask, -float("inf"))
            return logits

        for _ in range(max_tokens):
            # Build cache with full prompt once; then incremental
            if past_kvs is None:
                logits, past_kvs = self(generated)
            else:
                logits, past_kvs = self(generated[:, -1:], past_kvs)

            logits = logits[:, -1, :] / max(1e-5, float(temperature))
            # Dont sample the tags and especially pad/unk/bos
            logits[:, bad_ids] = -float("inf")

            # Repetition penalty (vectorized on unique tokens)
            if repetition_penalty is not None and repetition_penalty > 1.0:
                uniq = torch.unique(generated[0])
                logits[:, uniq] /= repetition_penalty

            # Top-k / top-p
            logits = filter_top_k_p(logits, top_k, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS or would overflow max_len
            if next_token.item() == 2: # EOS ID = 2
                break
            if generated.size(1) >= self.max_len:
                break

        return generated