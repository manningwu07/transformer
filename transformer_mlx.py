import math
import mlx.core as mx
import mlx.nn as nn
from params import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))

    def __call__(self, x):
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        t = mx.arange(max_seq_len).astype(mx.float32)
        freqs = mx.outer(t, freqs)
        self._cos = mx.cos(freqs)
        self._sin = mx.sin(freqs)

    def __call__(self, seq_len: int, offset: int = 0):
        return self._cos[offset : offset + seq_len], self._sin[offset : offset + seq_len]


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


class MLA(nn.Module):
    """
    DeepSeek V3 MLA with Decoupled RoPE for MLX.
    """

    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.d_model = args.d_model
        self.rope_dim = 64

        self.content_dim = self.head_dim - self.rope_dim

        # Query projections
        self.q_down = nn.Linear(args.d_model, args.q_lora_rank, bias=False)
        self.q_up = nn.Linear(args.q_lora_rank, args.n_heads * self.content_dim, bias=False)
        self.q_rope = nn.Linear(args.d_model, args.n_heads * self.rope_dim, bias=False)

        # KV projections
        self.kv_down = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.k_up = nn.Linear(args.d_latent, args.n_heads * self.content_dim, bias=False)
        self.v_up = nn.Linear(args.d_latent, args.n_heads * args.head_dim, bias=False)
        self.k_rope = nn.Linear(args.d_model, args.n_heads * self.rope_dim, bias=False)

        # Output
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.d_model, bias=False)

        self.rope = RotaryEmbedding(self.rope_dim, theta=args.rope_theta)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x, cache=None):
        B, T, C = x.shape

        pos_offset = cache["seq_len"] if cache is not None else 0

        # Q processing
        q_content = self.q_up(self.q_down(x))
        q_pos = self.q_rope(x)

        # KV processing
        kv_latent = self.kv_down(x)
        k_content = self.k_up(kv_latent)
        k_pos = self.k_rope(x)
        v = self.v_up(kv_latent)

        # Concatenate with cache
        if cache is not None:
            kv_latent = mx.concatenate([cache["latent"], kv_latent], axis=1)
            k_content = self.k_up(kv_latent)
            k_pos = mx.concatenate([cache["k_rope"], k_pos], axis=1)
            v = self.v_up(kv_latent)

        kv_len = kv_latent.shape[1]

        # Reshape for attention
        q_content = q_content.reshape(B, T, self.n_heads, self.content_dim).transpose(0, 2, 1, 3)
        q_pos = q_pos.reshape(B, T, self.n_heads, self.rope_dim).transpose(0, 2, 1, 3)
        k_content = k_content.reshape(B, kv_len, self.n_heads, self.content_dim).transpose(0, 2, 1, 3)
        k_pos = k_pos.reshape(B, kv_len, self.n_heads, self.rope_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, kv_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = self.rope(pos_offset + T)
        q_cos, q_sin = cos[pos_offset:], sin[pos_offset:]
        q_pos = apply_rope(q_pos, q_cos, q_sin)

        k_cos, k_sin = cos[:kv_len], sin[:kv_len]
        k_pos = apply_rope(k_pos, k_cos, k_sin)

        # Reassemble Q and K
        q = mx.concatenate([q_content, q_pos], axis=-1)
        k = mx.concatenate([k_content, k_pos], axis=-1)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask
        mask = mx.triu(mx.full((T, kv_len), -mx.inf), k=kv_len - T + 1)
        scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        new_cache = {
            "latent": kv_latent,
            "k_rope": k_pos.transpose(0, 2, 1, 3).reshape(B, kv_len, -1),
            "seq_len": kv_len,
        }

        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.attn = MLA(args)
        self.norm2 = RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.mlp = SwiGLU(args.d_model, args.hidden_size)

    def __call__(self, x, cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class LLM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else Config

        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.tok_embeddings.weight

    def __call__(self, idx, cache=None):
        x = self.tok_embeddings(idx)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, layer_cache)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.output(x)

        return logits, new_caches

    def generate(
        self,
        idx,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = None,
        eos_id: int = None,
        use_cache: bool = True,
    ):
        cache = None

        for _ in range(max_tokens):
            if cache is not None:
                idx_input = idx[:, -1:]
            else:
                idx_input = idx

            logits, cache = self(idx_input, cache=cache if use_cache else None)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                threshold = top_values[:, -1:]
                logits = mx.where(logits < threshold, mx.full(logits.shape, -mx.inf), logits)

            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
                sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
                probs = mx.softmax(sorted_logits, axis=-1)
                cum_probs = mx.cumsum(probs, axis=-1)

                mask = cum_probs > top_p
                # Keep at least one token
                mask = mx.concatenate([mx.zeros((mask.shape[0], 1), dtype=mx.bool_), mask[:, :-1]], axis=-1)
                sorted_logits = mx.where(mask, mx.full(sorted_logits.shape, -mx.inf), sorted_logits)
                logits = mx.take_along_axis(sorted_logits, mx.argsort(sorted_indices, axis=-1), axis=-1)

            # Sample
            probs = mx.softmax(logits, axis=-1)
            idx_next = mx.random.categorical(mx.log(probs + 1e-10))
            idx_next = idx_next.reshape(-1, 1)
            idx = mx.concatenate([idx, idx_next], axis=1)

            mx.eval(idx_next)  # Force evaluation for streaming

            if eos_id is not None and idx_next.item() == eos_id:
                break

            yield idx_next

        return idx