import math
import os
import random
import glob
import torch
import numpy as np
from typing import Any
import torch.distributed as dist
from typing import Dict, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional

from params import TrainCfg

def sanitize_state_dict_for_load(
    sd: Dict[str, Any],
    *,
    strip_orig_mod: bool = True,
    strip_rope_cache: bool = True,
) -> Dict[str, Any]:
    """
    Make checkpoints portable across:
      - torch.compile (keys prefixed with "_orig_mod.")
      - RoPE cached buffers that may be present in older checkpoints but are
        now non-persistent / recomputed (cos_cached, sin_cached).

    Use this in CLI / eval scripts before model.load_state_dict(...).
    """
    if sd is None:
        return sd

    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if strip_orig_mod and k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "", 1)

        if strip_rope_cache:
            # Drop any RoPE cached buffers anywhere in the module tree.
            # Matches e.g. "layers.0.attn.rope.cos_cached"
            if k.endswith("rope.cos_cached") or k.endswith("rope.sin_cached"):
                continue
            if k.endswith("cos_cached") or k.endswith("sin_cached"):
                # Safety: if you ever have other caches named similarly.
                # Comment these two lines out if you have non-RoPE buffers
                # with these names.
                continue

        out[k] = v
    return out


def load_model_state_from_checkpoint(
    ckpt_path: str,
    *,
    key: str = "model",
) -> Dict[str, Any]:
    """
    Load checkpoint and return a sanitized model state_dict.
    Works with:
      - raw state_dict checkpoints
      - CheckpointManager payloads {"model": ..., "optimizer": ..., ...}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get(key, ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"Expected state_dict dict, got: {type(sd)}")
    return sanitize_state_dict_for_load(sd)


def is_distributed():
    return int(os.getenv("WORLD_SIZE", "1")) > 1

def get_rank():
    return int(os.getenv("RANK", "0"))

def get_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))

def is_main_process():
    return get_rank() == 0

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    # Only grab CUDA state if we are in a process that can actually talk to the GPU
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state):
    if state is None: return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    
@torch.no_grad()
def validate_multi(
    model,
    device,
    loaders: Dict[str, object],
    max_val_steps: int = 100,
) -> Dict[str, Tuple[float, float]]:
    """
    Run validation separately for multiple loaders.
    Returns: {name: (avg_loss, ppl)}
    """
    out: Dict[str, Tuple[float, float]] = {}
    for name, loader in loaders.items():
        loss, ppl = validate(
            model=model,
            device=device,
            val_loader=loader,
            max_val_steps=max_val_steps,
        )
        out[name] = (loss, ppl)
    return out

@torch.no_grad()
def validate(model, device, val_loader, max_val_steps: int = 100):
    model.eval()
    total_loss = 0.0
    steps = 0

    for x, y in val_loader:
        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y)

        total_loss += float(loss.item())
        steps += 1
        if steps >= max_val_steps:
            break

    avg_loss = total_loss / max(1, steps)
    ppl = math.exp(avg_loss)
    model.train()
    return avg_loss, ppl


def make_scheduler(optimizer, total_opt_steps: int, warmup_steps: int, schedule: str):
    lr_start = float(TrainCfg.lr_start)
    lr_end = float(TrainCfg.lr_end)
    warmup_steps = int(warmup_steps)
    total_opt_steps = int(total_opt_steps)

    def lr_at(step: int) -> float:
        if total_opt_steps <= 1:
            return lr_end

        if warmup_steps > 0 and step < warmup_steps:
            t = step / max(1, warmup_steps)
            return lr_end + (lr_start - lr_end) * t

        t = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)

        if schedule == "linear":
            return lr_start + (lr_end - lr_start) * t

        # cosine
        return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * t))

    def lr_mult(step: int) -> float:
        return lr_at(step) / max(lr_start, 1e-12)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)

class CheckpointManager:
    def __init__(self, save_dir: str = "models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def _unwrap_model(self, model):
        # DDP wrapper
        if isinstance(model, DDP):
            model = model.module
        # torch.compile wrapper
        model = getattr(model, "_orig_mod", model)
        return model

    def _strip_orig_mod_prefix(self, sd: dict):
        # Normalize compiled checkpoints to eager key format
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        return sd

    def _strip_nonstrict_buffers(self, sd: dict):
        # Older checkpoints may contain RoPE cached buffers which we now mark
        # persistent=False. They are safe to drop.
        drop_suffixes = ("attn.rope.cos_cached", "attn.rope.sin_cached")
        out = {}
        for k, v in sd.items():
            if k.endswith(drop_suffixes):
                continue
            out[k] = v
        return out

    def save(
        self,
        tag,
        model,
        opt=None,
        sched=None,
        state=None,
        is_crash: bool = False,
        keep: int = 3,
        **kwargs: Any,
    ):
        # Back-compat with newer call sites
        opt = kwargs.get("optimizer", opt)
        sched = kwargs.get("scheduler", sched)
        state = kwargs.get("client_state", state)
        if not is_main_process(): return
        path = os.path.join(self.save_dir, f"{tag}.pt")
        tmp_path = path + ".tmp"
        m = self._unwrap_model(model)
        payload = {
            "model": m.state_dict(),
            "optimizer": opt.state_dict() if opt else None,
            "scheduler": sched.state_dict() if sched else None,
            "client_state": state,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        with open(os.path.join(self.save_dir, "latest"), "w") as f:
            f.write(tag)
        if not is_crash: self.prune(keep)

    def load(self, resume, model, opt=None, sched=None, **kwargs: Any):
        # Back-compat with newer call sites
        opt = kwargs.get("optimizer", opt)
        sched = kwargs.get("scheduler", sched)
        
        if resume is None:
            latest = os.path.join(self.save_dir, "latest")
            if not os.path.exists(latest): return None
            with open(latest, "r") as f: tag = f.read().strip()
            path = os.path.join(self.save_dir, f"{tag}.pt")
        else:
            path = resume if resume.endswith(".pt") else resume + ".pt"
        
        if is_main_process(): print(f"🔁 Loading: {path}")
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        m = self._unwrap_model(model)
        sd = self._strip_orig_mod_prefix(ckpt["model"])
        sd = self._strip_nonstrict_buffers(sd)
        m.load_state_dict(sd, strict=True)
        if opt and ckpt["optimizer"]: opt.load_state_dict(ckpt["optimizer"])
        if sched and ckpt["scheduler"]: sched.load_state_dict(ckpt["scheduler"])
        return ckpt.get("client_state")

    def prune(self, keep):
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, "*.pt")), 
                       key=os.path.getmtime)
        for p in ckpts[:-keep]:
            try: os.remove(p)
            except: pass