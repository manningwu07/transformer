import mlx.core as mx
import mlx.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        # In MLX, we don't need to wrap the base layer to freeze it; 
        # freezing is handled by which parameters are passed to the optimizer.
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Initialize LoRA weights
        # A: (in, r), B: (r, out)
        scale = 1 / math.sqrt(base_layer.weight.shape[1])
        self.A = mx.random.uniform(
            low=-scale, 
            high=scale, 
            shape=(base_layer.weight.shape[1], r)
        )
        self.B = mx.zeros((r, base_layer.weight.shape[0]))

    def __call__(self, x):
        # Forward pass: base(x) + (x @ A @ B) * scaling
        out_base = self.base(x)
        lora_out = (x @ self.A @ self.B) * self.scaling
        return out_base + lora_out