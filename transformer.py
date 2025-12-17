import math
import mlx.core as mx
import mlx.nn as nn
from lora.lora import LoRALinear
from params import Config

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=2048, lora_r=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # RoPE: Define rotary embedding
        self.rope = nn.RoPE(self.d_head, traditional=True)

        # Projections
        if lora_r:
            self.q_proj = LoRALinear(nn.Linear(d_model, d_model, bias=False), r=lora_r)
            self.v_proj = LoRALinear(nn.Linear(d_model, d_model, bias=False), r=lora_r)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, L, n_heads, d_head)
        q = q.reshape(B, L, self.n_heads, self.d_head)
        k = k.reshape(B, L, self.n_heads, self.d_head)
        v = v.reshape(B, L, self.n_heads, self.d_head)

        # Apply RoPE to Q and K
        # If we have a cache (inference), we need to offset the RoPE positions
        offset = cache[0].shape[1] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV Cache if present
        if cache is not None:
            past_k, past_v = cache
            k = mx.concatenate([past_k, k], axis=1)
            v = mx.concatenate([past_v, v], axis=1)
            cache = (k, v)

        # Flash Attention (Scaled Dot Product)
        # MLX expects (B, H, L, D) for transpose or handles it internally. 
        # mx.fast.scaled_dot_product_attention expects: (q, k, v, mask)
        # We need to swap dimensions to (B, H, L, D) for the fast kernel API usually
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply attention
        # Note: MLX scaled_dot_product_attention handles causal masking if a mask is provided,
        # but for causal specific implementation without explicit mask tensor, we rely on the mask passed in.
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1/math.sqrt(self.d_head))

        # Reassemble
        y = y.transpose(0, 2, 1, 3).reshape(B, L, D)
        
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y, cache

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x):
        x_gated = self.fc1(x)
        x_g, x_v = mx.split(x_gated, 2, axis=-1)
        # SwiGLU: SiLU(gate) * value
        return self.drop(self.fc2(nn.silu(x_g) * x_v))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_len):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.ln1(x)
        a, cache = self.attn(x, mask=mask, cache=cache)
        x = residual + a
        
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x, cache

class LLM(nn.Module):
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
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Note: Removed learned Positional Embeddings in favor of RoPE inside blocks
        
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_len)
            for _ in range(n_layers)
        ]
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

        # Weight Tying
        self.head.weight = self.tok_emb.weight

    def __call__(self, x, mask=None, cache=None):
        # x: (B, L)
        x = self.tok_emb(x)
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            c_i = cache[i] if cache is not None else None
            x, c_i = layer(x, mask=mask, cache=c_i)
            new_cache.append(c_i)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_cache

    def generate(self, idx, max_tokens=20, temp=0.7):
        # idx: (B, L) initial prompt
        y = idx
        cache = None

        # Create a causal mask for the initial pass
        # MLX fast attention supports additive mask. 
        # We generally rely on the implicit causal nature or pass a triangular mask.
        # For the prompt:
        L = y.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        mask = mask.astype(self.tok_emb.weight.dtype)

        # 1. Process Prompt
        logits, cache = self(y, mask=mask)
        
        # 2. Decode loop
        for _ in range(max_tokens):
            # Take last token logits
            last_logits = logits[:, -1, :] 
            
            # Simple Sampling
            if temp > 0:
                last_logits = last_logits / temp
                # MLX doesn't have Multinomial yet in all versions, 
                # but we can use categorical via random.categorical
                next_token = mx.random.categorical(last_logits, num_samples=1)
            else:
                next_token = mx.argmax(last_logits, axis=-1, keepdims=True)
            
            y = mx.concatenate([y, next_token], axis=1)
            
            if y.shape[1] >= self.max_len:
                break
                
            # Forward only the new token
            # Mask is None for single token (it attends to all past via cache)
            logits, cache = self(next_token, cache=cache)

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
            a = self.attn(self.ln1(x), past_kv)
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


class LLM(nn.Module):
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
        
        # Stabilize the initial logits (GPT does this too) -- prevents inital logits explosion
        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
    
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
        max_tokens: int = 20,
        top_k: int = 15,
        top_p: float = 0.9,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        bad_ids: list[int] | None = None,
        eos_id: int | None = None,
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

        # Set EOS from model if not provided
        if eos_id is None:
            eos_id = getattr(self, "eos_id", None)

        # --- Utility: filter logits ---
        def filter_top_k_p(logits, top_k=None, top_p=None):
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

            # Mask invalid/special tokens
            if bad_ids:
                logits[:, bad_ids] = -float("inf")

            # Apply repetition penalty correctly (handle sign)
            if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6:
                uniq_tokens = torch.unique(generated[0])
                for t in uniq_tokens:
                    val = logits[0, t]
                    if val > 0:
                        logits[0, t] = val / repetition_penalty
                    else:
                        logits[0, t] = val * repetition_penalty

            # Sampling filter
            logits = filter_top_k_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)

            # Degenerate fallback to greedy if NaNs/Inf
            if not torch.isfinite(probs).all() or (probs.sum(dim=-1) == 0).any():
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS/id overflow
            if eos_id is not None and next_token.item() == int(eos_id):
                break
            if generated.size(1) >= self.max_len:
                break

        return generated