import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base = base_layer  # original frozen linear layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Low-rank matrices (trainable)
        self.A = nn.Parameter(torch.randn(base_layer.in_features, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, base_layer.out_features))

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)  # frozen forward
        lora_update = (x @ self.A @ self.B) * self.scaling
        return out + lora_update