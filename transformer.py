import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import math
from params import Config
import inspect
from fused_swiglu_mlp import FusedSwiGLUMLP

class RMSNorm(nn.Module):
    """
    Custom RMSNorm to avoid PyTorch's fused RMSNorm kernels that can produce
    oversized Triton shared-mem requirements on some builds/GPUs.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=Config.max_seq_len, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

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
        max_seq_len = int(getattr(args, "max_seq_len", Config.max_seq_len))
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
                q_latent,
                lat_k.transpose(-1, -2),
            )
            # scores_pos: [B,H,T,kv_len]
            scores_pos = torch.matmul(
                q_pos,
                k_pos.transpose(-1, -2),
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

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.attn = MLA(args)
        self.norm2 = RMSNorm(args.d_model, eps=args.rms_norm_eps)

        self.mlp = FusedSwiGLUMLP(
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            bias=False,
        )

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

    def forward_no_cache(self, x):
        """
        Training/checkpoint-friendly forward. No KV cache, returns only x.
        This is what we compile and what checkpoint() calls.
        """
        attn_out, _ = self.attn(self.norm1(x), kv_cache=None, use_cache=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

    # Alias for backward compat (your checkpoint calls this)
    forward_train = forward_no_cache


# Fused cross-entropy (avoids materializing huge logit tensor)
try:
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyLoss
    _fused_ce = LigerCrossEntropyLoss()
    USE_FUSED_CE = True
except ImportError:
    USE_FUSED_CE = False

class LLM(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config if config is not None else Config
        
        # Allow overriding config with kwargs
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        self.max_seq_len = getattr(self.config, 'max_seq_len', Config.max_seq_len)
        
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        
         # Build layers - optionally compile each block individually
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) 
            for _ in range(self.config.n_layers)
        ])
        
        self._maybe_enable_torchao_float8()

        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        
        # Init weights, then tie (output shares embedding weights)
        self.apply(self._init_weights)
        self.output.weight = self.tok_embeddings.weight
        print(
            f"🧠 Model Init: MLA Decoupled RoPE | "
            f"Dim {self.config.d_model} | Latent {self.config.d_latent}"
        )
        
        self._apply_compilation()

    def _apply_compilation(self):
        mode = getattr(self.config, "compile_mode", "none")
        
        if mode == "none":
            return
        
        if mode == "layers":
            print("⚡ Compiling TransformerBlock.forward_no_cache (per-layer)...")
            for layer in self.layers:
                layer.forward_no_cache = torch.compile(
                    layer.forward_no_cache,
                    mode="default",
                    fullgraph=False,
                )
                layer.forward_train = layer.forward_no_cache
        
        elif mode == "model":
            # NOTE: We don't compile here — we return self and let train.py wrap it
            # This is because compile should happen AFTER .to(device) and before DDP
            print("⚡ Model marked for whole-model compilation (apply in train.py)")
            self._needs_whole_model_compile = True
        else:
            raise ValueError(f"Unknown compile_mode: {mode}")

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

        # checkpoint all layers except every Nth layer (skip to save compute)
        skip_every_n = getattr(self.config, "checkpoint_skip_every_n", 0)

        for i, layer in enumerate(self.layers):
            if use_ckpt and (skip_every_n == 0 or (i % skip_every_n) != 0):
                x = ckpt.checkpoint(layer.forward_no_cache, x, use_reentrant=False)
            else:
                x = layer.forward_no_cache(x)

        x = self.norm(x)
        logits = self.output(x)

        if targets is not None:
            if USE_FUSED_CE:
                # Fused CE: doesn't materialize [B*T, V] intermediate
                loss = _fused_ce(logits.view(-1, self.config.vocab_size), targets.view(-1))
            else:
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
            
    def _maybe_enable_torchao_float8(self) -> None:
        if not getattr(self.config, "use_float8", False):
            return

        try:
            from torchao.float8 import convert_to_float8_training
        except Exception as e:
            raise RuntimeError(
                "use_float8=True but torchao.float8 is not available. "
                "Install torchao and ensure it matches your torch build."
            ) from e

        # Optional: try to pass a Float8LinearConfig if the torchao version supports it.
        cfg = None
        try:
            from torchao.float8 import Float8LinearConfig

            recipe = getattr(self.config, "float8_recipe", "rowwise")
            sig = inspect.signature(Float8LinearConfig)
            kwargs = {}
            if "recipe_name" in sig.parameters:
                kwargs["recipe_name"] = recipe
            cfg = Float8LinearConfig(**kwargs) if kwargs else Float8LinearConfig()
        except Exception:
            cfg = None

        # Convert only the transformer layers (keeps tok_embeddings  lm_head BF16).
        convert_sig = inspect.signature(convert_to_float8_training)
        kwargs = {}
        if "config" in convert_sig.parameters and cfg is not None:
            kwargs["config"] = cfg

        out = convert_to_float8_training(self.layers, **kwargs)
        if out is not None:
            self.layers = out