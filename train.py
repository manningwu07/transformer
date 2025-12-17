import time
import math
import torch
import torch.nn as nn
import os
import argparse
import signal
import sys
import json
import traceback
import glob
from transformers import Adafactor
from torch.utils.data import DataLoader
from dataset import IndexedBinaryDataset

# Imports from your project
from transformer import LLM
from params import Config, TrainCfg

# --- Optimizations ---
torch.backends.cuda.matmul.allow_tf32 = True # Huge speedup on Ampere+
torch.backends.cudnn.benchmark = True

# --- Scheduler (Cosine with Warmup) ---
def get_lr(step, cfg, total_steps):
    # Warmup
    if step < cfg.warmup_steps:
        return cfg.lr_start * (step / cfg.warmup_steps)
    
    # Cosine Decay
    # Decay from lr_start to lr_end over the remaining steps
    decay_steps = total_steps - cfg.warmup_steps
    step_in_decay = min(step - cfg.warmup_steps, decay_steps)
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay / decay_steps))
    decayed = (cfg.lr_start - cfg.lr_end) * cosine_decay + cfg.lr_end
    return decayed

# --- Checkpoint & Logger (Same as before, condensed) ---
class CheckpointManager:
    def __init__(self, model, optimizer, save_dir="models"):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    def save(self, step, loss, is_crash=False):
        prefix = "crash_" if is_crash else "ckpt_"
        path = os.path.join(self.save_dir, f"{prefix}step_{step}.pt")
        state = {
            "step": step, "loss": loss, "config": Config.__dict__,
            "model": self.model.state_dict(), "optim": self.optimizer.state_dict()
        }
        try:
            torch.save(state, path)
            print(f"💾 Saved: {path}")
            if not is_crash: self._prune()
        except Exception as e: print(f"❌ Save Failed: {e}")
    def load(self, path):
        print(f"resuming from {path}...")
        c = torch.load(path, map_location="cpu")
        self.model.load_state_dict(c["model"])
        self.optimizer.load_state_dict(c["optim"])
        return c["step"]
    def _prune(self):
        files = sorted(glob.glob(os.path.join(self.save_dir, "ckpt_step_*.pt")), key=os.path.getmtime)
        for f in files[:-3]: os.remove(f)
        
class LossTracker:
    """Tracks loss for spike detection and NaN handling."""
    def __init__(self, window_size=100, spike_threshold=3.0):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.history = []
        
    def update(self, loss_val):
        self.history.append(loss_val)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    @property
    def running_avg(self):
        if not self.history:
            return float('inf')
        return sum(self.history) / len(self.history)
    
    def is_spike(self, loss_val):
        if len(self.history) < 50:
            return False
        return loss_val > self.spike_threshold * self.running_avg
    
    def is_valid(self, loss_tensor):
        return torch.isfinite(loss_tensor).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/shards")
    args = parser.parse_args()

    # 1. Init Model
    print(f"🚀 Initializing Model [MLA + SwiGLU]...")
    model = LLM(Config).cuda().to(dtype=torch.bfloat16)
    
    # OPTIMIZATION: Torch Compile
    # This fuses layers and optimizes the graph. 
    # Note: First step will be slow (compiling), then it flies.
    try:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model) 
    except Exception as e:
        print(f"⚠️ Compile failed: {e}. Running in eager mode.")

    optimizer = Adafactor(
        model.parameters(),
        lr=TrainCfg.lr_start,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False
    )
    
    ckpt = CheckpointManager(model, optimizer)
    start_step = 0
    if args.resume: start_step = ckpt.load(args.resume)

    # 2. Data Loader (Binary)
    print(f"📂 Loading Binary Dataset...")
    # NOTE: Ensure you ran prepare_data.py first!
    try:
        train_ds = IndexedBinaryDataset(args.data_dir, split="train", seq_len=TrainCfg.seq_len)
    except FileNotFoundError:
        print(f"❌ Error: No shards found at {args.data_dir}/train-*.bin")
        print("   Run: python3 scripts/makeDatasetFast.py")
        return

    train_loader = DataLoader(
        train_ds, 
        batch_size=TrainCfg.batch_size, 
        num_workers=4,          # Keep CPU ahead of GPU
        pin_memory=True,        # Faster transfer to CUDA
        persistent_workers=True,# Don't kill workers between epochs
        shuffle=True
    )
    
    # Calculate Total Steps (Approx)
    total_tokens = len(train_ds) * TrainCfg.seq_len
    tokens_per_step = TrainCfg.batch_size * TrainCfg.seq_len * TrainCfg.grad_accum_steps
    total_steps = int(total_tokens / tokens_per_step * 3) # Approx 3 epochs? Or set manually.
    total_steps = max(total_steps, 200_000) # Minimum baseline

    # 3. Training Loop
    model.train()
    step = start_step
    loss_tracker = LossTracker(window_size=100, spike_threshold=3.0)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n🛑 Saving & Exiting...")
        ckpt.save(step, 0.0, is_crash=True)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    iter_loader = iter(train_loader)
    accum_loss = 0.0
    t0 = time.time()
    
    print(f"🔥 Training Start: {total_steps} steps total")

    while step < total_steps:
        try:
            # Get Batch
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                x, y = next(iter_loader)

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            
            # Forward
            # Note: We rely on internal loss calculation or do it here
            logits, loss, _ = model(x, targets=y)
            
            # NaN/Inf check
            if not loss_tracker.is_valid(loss):
                print(f"⚠️ Step {step}: Non-finite loss detected, skipping batch")
                optimizer.zero_grad()
                continue
            
            raw_loss = loss.item()
            
            # Spike detection
            if loss_tracker.is_spike(raw_loss):
                print(f"⚠️ Step {step}: Loss spike detected ({raw_loss:.4f} vs avg {loss_tracker.running_avg:.4f})")
                # Option: skip, reduce LR, or just warn
            
            loss_tracker.update(raw_loss)

            loss = loss / TrainCfg.grad_accum_steps
            loss.backward()
            accum_loss += raw_loss  # Track true loss, not scaled
            
            # Step
            if (step + 1) % TrainCfg.grad_accum_steps == 0:
                # Scheduler
                lr = get_lr(step, TrainCfg, total_steps)
                for pg in optimizer.param_groups: pg['lr'] = lr
                
                # Clip & Update
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                
                if step % 10 == 0:
                    dt = time.time() - t0
                    tps = (TrainCfg.batch_size * TrainCfg.seq_len * TrainCfg.grad_accum_steps * 10) / dt
                    avg_loss = accum_loss / TrainCfg.grad_accum_steps
                    print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {tps:.0f} tok/s")
                    t0 = time.time()
                    accum_loss = 0.0

                if step % 1000 == 0:
                    ckpt.save(step, 0.0)

        except Exception as e:
            print(f"💥 Crash: {e}")
            traceback.print_exc()
            ckpt.save(step, 0.0, is_crash=True)
            break

if __name__ == "__main__":
    main()