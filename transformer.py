import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import math
from params import Config, TrainCfg

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=32768, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len, ...].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:seq_len, ...].to(device=x.device, dtype=x.dtype)
        return cos, sin

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)

class MLA(nn.Module):
    """
    Corrected DeepSeek V3 MLA with Decoupled RoPE.
    Splits content (compressed) from position (RoPE).
    """
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.d_model = args.d_model
        self.rope_dim = 64 # Fixed decoupled RoPE dim
        
        # Content Head Dim (Total head dim - RoPE part)
        self.content_dim = self.head_dim - self.rope_dim
        
        # 1. Query Projections
        # Content Part (Compressed)
        self.q_down = nn.Linear(args.d_model, args.q_lora_rank, bias=False)
        self.q_up = nn.Linear(args.q_lora_rank, args.n_heads * self.content_dim, bias=False)
        # RoPE Part (Not Compressed)
        self.q_rope = nn.Linear(args.d_model, args.n_heads * self.rope_dim, bias=False)

        # 2. KV Projections
        # Latent Content (The compressed part you cache)
        self.kv_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        
        # Up-Projections (Reconstruct K and V from latent)
        self.k_up = nn.Linear(args.d_latent, args.n_heads * self.content_dim, bias=False)
        self.v_up = nn.Linear(args.d_latent, args.n_heads * args.head_dim, bias=False) # V is full head dim
        
        # RoPE Part for K (Not Compressed)
        self.k_rope = nn.Linear(args.d_model, args.n_heads * self.rope_dim, bias=False)

        # 3. Output
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.d_model, bias=False)
        max_seq_len = int(getattr(args, "max_seq_len", 32768))
        self.rope = RotaryEmbedding(
            self.rope_dim,
            max_seq_len=max_seq_len,
            theta=args.rope_theta,
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        
        pos_offset = kv_cache["seq_len"] if kv_cache is not None else 0
        
        # --- Q Processing ---
        q_content = self.q_up(self.q_down(x))       # [B, T, H * content_dim]
        q_pos = self.q_rope(x)                      # [B, T, H * rope_dim]
        
         # --- KV Processing ---
        kv_latent_new = self.kv_down(x)  # [B, T, d_latent]
        k_pos_new = self.k_rope(x)       # [B, T, H * rope_dim] (pre-RoPE)

        if kv_cache is not None:
            kv_latent = torch.cat([kv_cache["latent"], kv_latent_new], dim=1)
        else:
            kv_latent = kv_latent_new
        
        # Reshape for Attention
        q_content = q_content.view(B, T, self.n_heads, self.content_dim).transpose(1, 2)
        q_pos = q_pos.view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)
        
        kv_len = kv_latent.size(1)

        # Apply RoPE (Only to the position part)
        # We want RoPE for absolute positions [0..kv_len)
        cos, sin = self.rope(q_pos, kv_len)

        # Q RoPE for the query positions [pos_offset..pos_offset+T)
        q_cos = cos[pos_offset:pos_offset + T]
        q_sin = sin[pos_offset:pos_offset + T]
        q_pos = apply_rope(q_pos, q_cos, q_sin)
        
        # K RoPE: cache should store *already roped* keys for O(t)
        k_pos_new = k_pos_new.view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)
        k_cos_new = cos[pos_offset:pos_offset + T]
        k_sin_new = sin[pos_offset:pos_offset + T]
        k_pos_new = apply_rope(k_pos_new, k_cos_new, k_sin_new)  # [B,H,T,rope_dim]

        if kv_cache is not None:
            k_pos = torch.cat([kv_cache["k_rope"], k_pos_new], dim=2)  # [B,H,kv_len,rope_dim]
        else:
            k_pos = k_pos_new  # [B,H,kv_len,rope_dim]

        # =========================
        # Inference fast path: small cache  O(t) compute
        # =========================
        if use_cache:
            # Project Q-content into latent space using k_up weights per head.
            # k_up.weight: [H*content_dim, d_latent] -> [H, content_dim, d_latent]
            Wk = self.k_up.weight.view(self.n_heads, self.content_dim, -1)  # [H, Cc, Dl]
            # q_latent = q_content @ Wk  -> [B,H,T,d_latent]
            q_latent = torch.einsum("bhtc,hcd->bhtd", q_content, Wk)

            # latent keys shared across heads: [B,kv_len,d_latent] -> [B,1,kv_len,d_latent]
            lat_k = kv_latent.unsqueeze(1)
            # scores_content: [B,H,T,kv_len]
            scores_content = torch.matmul(
                q_latent.to(torch.float32),
                lat_k.to(torch.float32).transpose(-1, -2),
            )
            # scores_pos: [B,H,T,kv_len]
            scores_pos = torch.matmul(
                q_pos.to(torch.float32),
                k_pos.to(torch.float32).transpose(-1, -2),
            )

            scores = (scores_content + scores_pos) * self.scale

            # Causal mask (needed for prefill when T>1)
            # forbid keys with index > pos_offset  query_index
            if T > 1 or (kv_cache is None and kv_len == T):
                mask = torch.full(
                    (T, kv_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                mask = torch.triu(mask, diagonal=pos_offset + 1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)

            attn = torch.softmax(scores, dim=-1).to(q_latent.dtype)  # [B,H,T,kv_len]

            # latent_context = sum_t attn * latent_t  -> [B,H,T,d_latent]
            latent_context = torch.matmul(attn, lat_k)

            # Apply v_up after the weighted sum (linearity):
            # v_up.weight: [H*head_dim, d_latent] -> [H, head_dim, d_latent]
            Wv = self.v_up.weight.view(self.n_heads, self.head_dim, -1)  # [H, Hd, Dl]
            out = torch.einsum("bhtd,hmd->bhtm", latent_context, Wv)

            out = out.transpose(1, 2).contiguous().view(B, T, -1)

            new_cache = {
                "latent": kv_latent.detach(),
                "k_rope": k_pos.detach(),  # [B,H,kv_len,rope_dim] roped
                "seq_len": kv_len,
            }
            return self.o_proj(out), new_cache

        # =========================
        # Training path: keep SDPA (FlashAttention) behavior
        # =========================

        # Recompute full K-content and V for training (uses fast SDPA kernels)
        k_content = self.k_up(kv_latent).view(B, kv_len, self.n_heads, self.content_dim).transpose(1, 2)
        v = self.v_up(kv_latent).view(B, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # k_pos currently has shape [B,H,kv_len,rope_dim] already
        q = torch.cat([q_content, q_pos], dim=-1)          # [B,H,T,head_dim]
        k = torch.cat([k_content, k_pos], dim=-1)          # [B,H,kv_len,head_dim]

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), None

class SwiGLU_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.d_model, args.hidden_size, bias=False)
        self.w2 = nn.Linear(args.hidden_size, args.d_model, bias=False)
        self.w3 = nn.Linear(args.d_model, args.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.attn = MLA(args)
        self.norm2 = RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.mlp = SwiGLU_MLP(args)

    def forward(self, x, kv_cache=None, use_cache=False, use_checkpoint=False):
        if use_checkpoint and self.training:
            # Checkpoint ONLY the MLA part (the biggest activation hog)
            def create_custom_forward(module):
                def custom_forward(inputs):
                    out, _ = module(inputs, None, False)
                    return out
                return custom_forward
            attn_out = ckpt.checkpoint(create_custom_forward(self.attn), self.norm1(x), use_reentrant=False)
            new_cache = None
        else:
            attn_out, new_cache = self.attn(self.norm1(x), kv_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache

    def forward_train(self, x):
        # Training path: no KV cache, returns only x (checkpoint-friendly)
        attn_out, _ = self.attn(self.norm1(x), kv_cache=None, use_cache=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class LLM(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config if config is not None else Config
        
        # Allow overriding config with kwargs
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        self.max_seq_len = getattr(self.config, 'max_seq_len', 16384)
        
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        
         # Build layers - optionally compile each block individually
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) 
            for _ in range(self.config.n_layers)
        ])
        
        # Compile the inner forward_train method (what gets checkpointed)
        # This way: checkpointing controls memory, compile speeds up the ops inside
        self._compile_layers = getattr(self.config, "compile_layers", False)
        if self._compile_layers:
            print("⚡ Compiling TransformerBlock.forward_train (compatible with grad ckpt)...")
            for layer in self.layers:
                layer.forward_train = torch.compile(
                    layer.forward_train,
                    mode="default",
                    fullgraph=False,
                )

        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        
        # Init weights, then tie (output shares embedding weights)
        self.apply(self._init_weights)
        self.output.weight = self.tok_embeddings.weight
        print(
            f"🧠 Model Init: MLA Decoupled RoPE | "
            f"Dim {self.config.d_model} | Latent {self.config.d_latent}"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None, use_cache: bool = False):
        x = self.tok_embeddings(idx)

        use_ckpt = (
            self.training
            and targets is not None
            and getattr(self.config, "gradient_checkpointing", False)
            and not use_cache
        )
        
        checkpoint_every_n = getattr(self.config, "checkpoint_every_n", 1)

        # Every 'n' layers we will NOT checkpoint to save compute
        # 1 = checkpoint everything (current)
        # 2 = checkpoint every 2nd layer (faster)
        for i, layer in enumerate(self.layers):
            if use_ckpt:
                if (i % checkpoint_every_n == 0):
                    x, _ = layer(x, None, False)
                else:
                    x = ckpt.checkpoint(layer.forward_train, x, use_reentrant=False)
            else:
                x, _ = layer(x, None, False)

        x = self.norm(x)
        logits = self.output(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )
            return logits, loss, None
        
        return logits, None, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.7, top_k=50, use_cache=True):
        """
        Autoregressive generation with optional KV caching.
        use_cache=True: O(n) per token (fast)
        use_cache=False: O(n²) per token (slow, but simpler)
        """
        kv_cache = None


        for _ in range(max_new_tokens):
            if use_cache and kv_cache is not None:
                # Only process the last token
                idx_input = idx[:, -1:]
            else:
                # Process full sequence (first pass or no cache)
                idx_input = idx[:, -self.max_seq_len:]
            
            logits, _, kv_cache = self(idx_input, use_cache=use_cache, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            yield idx_next # Yield token-by-token for streaming