# transformer.py
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
        # Precompute RoPE frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x, seq_len):
        # x: [Batch, Heads, Seq, HeadDim]
        # Return cos, sin for the relevant sequence length
        return self.cos_cached[:seq_len, ...].to(x.device), \
               self.sin_cached[:seq_len, ...].to(x.device)

def apply_rope(x, cos, sin):
    # Rotate standard Q/K vectors
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek V3 Style)
    Compresses KV into a latent vector to save VRAM and Bandwidth.
    """
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.d_model = args.d_model
        
        # 1. Query Compression (Model -> Latent -> Heads)
        self.q_down = nn.Linear(args.d_model, args.q_lora_rank, bias=False)
        self.q_up = nn.Linear(args.q_lora_rank, args.n_heads * args.head_dim, bias=False)
        
        # 2. KV Compression (Model -> Latent -> Heads)
        # This 'kv_down' output is what gets cached during inference! (512 dims vs 2048)
        self.kv_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.kv_up = nn.Linear(args.d_latent, args.n_heads * args.head_dim, bias=False)
        
        # 3. Output Projection
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.d_model, bias=False)
        
        self.rope = RotaryEmbedding(args.head_dim, theta=args.rope_theta)

    def forward(self, x):
        B, T, C = x.shape
        
        # --- Q Processing ---
        # Compress -> Decompress
        q_latent = self.q_down(x)
        q = self.q_up(q_latent)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, T, D]
        
        # --- KV Processing (MLA Magic) ---
        # Compress (This 512-dim vector is what you'd cache in inference)
        kv_latent = self.kv_down(x) 
        
        # Decompress (Reconstruct full heads for calculation)
        # Note: In pure inference, we can merge this w/ Query, but for training, 
        # reconstructing K/V is cleaner for FlashAttn kernels.
        kv = self.kv_up(kv_latent)
        kv = kv.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, T, D]
        
        k, v = kv, kv  # In simple MLA, K and V share the same projection
        
        # --- RoPE ---
        cos, sin = self.rope(q, T)
        # Apply RoPE to Q and K
        # (DeepSeek technically uses a decoupled strategy, but standard RoPE 
        # on the up-projected heads is mathematically stable for 1B scale)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # --- Flash Attention ---
        # Uses standard PyTorch optimized kernel (FlashAttn v2 backend usually)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

class SwiGLU_MLP(nn.Module):
    """
    Standard High-Performance MLP (Llama style).
    Replaces the MoE. Dense, robust, harder to mess up.
    """
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.d_model, args.hidden_size, bias=False) # Gate
        self.w2 = nn.Linear(args.hidden_size, args.d_model, bias=False) # Down
        self.w3 = nn.Linear(args.d_model, args.hidden_size, bias=False) # Up

    def forward(self, x):
        # F.silu is Swish
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
    def __init__(self):
        super().__init__()
        self.config = Config
        
        self.tok_embeddings = nn.Embedding(Config.vocab_size, Config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(Config) for _ in range(Config.n_layers)])
        self.norm = RMSNorm(Config.d_model, eps=Config.rms_norm_eps)
        self.output = nn.Linear(Config.d_model, Config.vocab_size, bias=False)
        
        # Weight Tying (Standard practice for small models)
        self.tok_embeddings.weight = self.output.weight
        
        self.apply(self._init_weights)
        print(f"🧠 Model Initialized: 1B Params | MLA (512 lat) | Context: {TrainCfg.seq_len}")

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
            # Training Mode (Return Loss)
            logits = self.output(x)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            return logits, loss
        else:
            # Inference Mode (Return just logits for last token)
            logits = self.output(x[:, [-1], :])
            return logits, None