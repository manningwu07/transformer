# muon.py
"""
MUON: MomentUm Orthogonalized by Newton-schulz
Based on Keller Jordan's implementation from modded-nanogpt
"""
import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the matrix square root inverse.
    Orthogonalizes G in-place (approximately).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)

    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    MUON optimizer.
    
    2D+ params: Newton-Schulz orthogonalized momentum (MUON)
    1D params: AdamW fallback
    
    Args:
        lr: MUON learning rate (default: 0.02)
        momentum: momentum coefficient (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)
        adamw_lr: AdamW LR for 1D params (default: 3e-4)
        adamw_betas: AdamW betas (default: (0.9, 0.95))
        adamw_wd: AdamW weight decay (default: 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_wd: float = 0.01,
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adamw_lr = group["adamw_lr"]
            beta1, beta2 = group["adamw_betas"]
            adamw_wd = group["adamw_wd"]
            adamw_eps = group["adamw_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # 2D+ params: MUON
                if p.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        update = g + momentum * buf
                    else:
                        update = buf.clone()

                    # Flatten to 2D for Newton-Schulz
                    orig_shape = update.shape
                    if update.ndim > 2:
                        update = update.view(update.size(0), -1)

                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    update = update.view(orig_shape)

                    # Scale: Keller uses sqrt(max(fan_in, fan_out))
                    scale = max(p.size(0), p.size(-1)) ** 0.5
                    p.add_(update, alpha=-lr * scale)

                # 1D params: AdamW
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    # Decoupled weight decay
                    p.mul_(1.0 - adamw_lr * adamw_wd)

                    exp_avg.lerp_(g, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]

                    p.addcdiv_(
                        exp_avg,
                        exp_avg_sq.sqrt().add_(adamw_eps),
                        value=-adamw_lr / bc1 * (bc2**0.5),
                    )

        return loss