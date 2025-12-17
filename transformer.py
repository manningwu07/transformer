import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from params import Config, TrainCfg

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=16384, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len, ...].to(x.device), \
               self.sin_cached[:seq_len, ...].to(x.device)

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
        
        self.rope = RotaryEmbedding(self.rope_dim, theta=args.rope_theta)

    def forward(self, x):
        B, T, C = x.shape
        
        # --- Q Processing ---
        q_content = self.q_up(self.q_down(x))       # [B, T, H * content_dim]
        q_pos = self.q_rope(x)                      # [B, T, H * rope_dim]
        
        # --- KV Processing ---
        kv_latent = self.kv_down(x)                 # [B, T, d_latent] <-- CACHE THIS IN INFERENCE
        k_content = self.k_up(kv_latent)            # [B, T, H * content_dim]
        k_pos = self.k_rope(x)                      # [B, T, H * rope_dim] <-- AND THIS
        v = self.v_up(kv_latent)                    # [B, T, H * head_dim]
        
        # Reshape for Attention
        q_content = q_content.view(B, T, self.n_heads, self.content_dim).transpose(1, 2)
        q_pos = q_pos.view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)
        k_content = k_content.view(B, T, self.n_heads, self.content_dim).transpose(1, 2)
        k_pos = k_pos.view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (Only to the position part)
        cos, sin = self.rope(q_pos, T)
        q_pos = apply_rope(q_pos, cos, sin)
        k_pos = apply_rope(k_pos, cos, sin)
        
        # Re-assemble Q and K (Content + Position)
        q = torch.cat([q_content, q_pos], dim=-1)   # [B, H, T, head_dim]
        k = torch.cat([k_content, k_pos], dim=-1)   # [B, H, T, head_dim]
        
        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
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
        
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        
        self.tok_embeddings.weight = self.output.weight
        self.apply(self._init_weights)
        print(f"🧠 Model Init: MLA Decoupled RoPE | Dim {self.config.d_model} | Latent {self.config.d_latent}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        if targets is not None:
            logits = self.output(x)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            return logits, loss
        else:
            logits = self.output(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.7, top_k=50):
        # Stateless generation loop (simple but effective for testing)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.d_model:] # Safety crop
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            yield idx_next # Yield token-by-token for streaming