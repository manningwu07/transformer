import math
import os
import random
import glob
import torch
import numpy as np
from typing import Any
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from params import TrainCfg

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
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }

def set_rng_state(state):
    if state is None: return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    

@torch.no_grad()
def validate(model, device, val_loader, max_val_steps: int = 100):
    model.eval()
    total_loss = 0.0
    steps = 0

    for x, y in val_loader:
        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y, return_logits=False)

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
        payload = {
            "model": model.module.state_dict() if isinstance(model, DDP) 
                     else model.state_dict(),
            "optimizer": opt.state_dict() if opt else None,
            "scheduler": sched.state_dict() if sched else None,
            "client_state": state,
        }
        torch.save(payload, path)
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
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if opt and ckpt["optimizer"]: opt.load_state_dict(ckpt["optimizer"])
        if sched and ckpt["scheduler"]: sched.load_state_dict(ckpt["scheduler"])
        return ckpt.get("client_state")

    def prune(self, keep):
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, "*.pt")), 
                       key=os.path.getmtime)
        for p in ckpts[:-keep]:
            try: os.remove(p)
            except: pass
            
            