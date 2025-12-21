import time
import math
import torch
import torch.nn as nn
import os
import argparse
import signal
import sys
import traceback
import glob
from transformers import Adafactor
from torch.utils.data import DataLoader
from dataset import IndexedBinaryDataset

# Imports from your project
from params import Config, TrainCfg

# --- Backend Selection Logic ---
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--data_dir", type=str, default="data/shards")
parser.add_argument(
    "--backend", 
    type=str, 
    default="mlx" if sys.platform == "darwin" else "torch",
    choices=["torch", "mlx"],
    help="Force a specific backend. Defaults to mlx on macOS."
)
args, _ = parser.parse_known_args()

if args.backend == "mlx":
    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as mopt
    from transformer_mlx import LLM
    print("🍏 Using MLX Backend (Apple Silicon Optimized)")
else:
    from transformer import LLM
    print("🔥 Using PyTorch Backend")

# --- Optimizations (Torch Only) ---
if args.backend == "torch":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# --- Scheduler (Cosine with Warmup) ---
def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr_start * (step / cfg.warmup_steps)
    
    decay_steps = total_steps - cfg.warmup_steps
    step_in_decay = min(step - cfg.warmup_steps, decay_steps)
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay / decay_steps))
    decayed = (cfg.lr_start - cfg.lr_end) * cosine_decay + cfg.lr_end
    return decayed

# --- Checkpoint & Logger ---
class CheckpointManager:
    def __init__(self, model, optimizer, save_dir="models"):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, step, loss, is_crash=False):
        prefix = "crash_" if is_crash else "ckpt_"
        path = os.path.join(self.save_dir, f"{prefix}step_{step}.pt")
        
        if args.backend == "torch":
            state = {
                "step": step, "loss": loss, "config": Config.__dict__,
                "model": self.model.state_dict(), "optim": self.optimizer.state_dict()
            }
            torch.save(state, path)
        else:
            # MLX Save logic (uses .safetensors or .npz style)
            self.model.save_weights(os.path.join(self.save_dir, f"{prefix}step_{step}.safetensors"))
            
        print(f"💾 Saved: {path}")
        if not is_crash: self._prune()

    def load(self, path):
        print(f"resuming from {path}...")
        if args.backend == "torch":
            c = torch.load(path, map_location="cpu")
            self.model.load_state_dict(c["model"])
            self.optimizer.load_state_dict(c["optim"])
            return c["step"]
        else:
            self.model.load_weights(path)
            return 0 # Implementation dependent

    def _prune(self):
        pattern = "*.pt" if args.backend == "torch" else "*.safetensors"
        files = sorted(glob.glob(os.path.join(self.save_dir, f"ckpt_step_{pattern}")), key=os.path.getmtime)
        for f in files[:-3]: os.remove(f)
        
class LossTracker:
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
        if not self.history: return float('inf')
        return sum(self.history) / len(self.history)
    
    def is_spike(self, loss_val):
        if len(self.history) < 50: return False
        return loss_val > self.spike_threshold * self.running_avg
    
    def is_valid(self, loss_val):
        return math.isfinite(loss_val)

def main():
    # 1. Init Model
    if args.backend == "torch":
        model = LLM(Config).cuda().to(dtype=torch.bfloat16)
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
    else:
        # MLX Initialization
        model = LLM(Config)
        mx.eval(model.parameters())
        optimizer = mopt.Adafactor(learning_rate=TrainCfg.lr_start)

    ckpt = CheckpointManager(model, optimizer)
    start_step = 0
    if args.resume: start_step = ckpt.load(args.resume)

    # 2. Data Loader
    print(f"📂 Loading Binary Dataset from {args.data_dir}...")
    train_ds = IndexedBinaryDataset(args.data_dir, split="train", seq_len=TrainCfg.seq_len)
    
    # We use torch DataLoader even for MLX but convert tensors in the loop
    train_loader = DataLoader(
        train_ds, 
        batch_size=TrainCfg.batch_size, 
        num_workers=0 if args.backend == "mlx" else 4, # MLX likes 0 for simpler memory
        pin_memory=(args.backend == "torch"),
        persistent_workers=(args.backend == "torch" and args.backend != "mlx"),
        shuffle=True
    )
    
    total_steps = max(int(len(train_ds) / (TrainCfg.batch_size * TrainCfg.grad_accum_steps)), 200_000)
    loss_tracker = LossTracker()
    
    def signal_handler(sig, frame):
        print("\n🛑 Saving & Exiting...")
        ckpt.save(step, 0.0, is_crash=True)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    step = start_step
    accum_loss = 0.0
    t0 = time.time()
    
    print(f"🔥 Training Start: {total_steps} steps total")

    # --- MLX Training Step Function ---
    if args.backend == "mlx":
        def loss_fn(model, x, y):
            logits, _ = model(x)
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            return mx.mean(mnn.losses.cross_entropy(logits, y))
        
        state = [model.state, optimizer.state]
        @mx.compile
        def train_step(x, y):
            loss_and_grad_fn = mnn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            return loss

    # --- Unified Loop ---
    while step < total_steps:
        for x, y in train_loader:
            try:
                lr = get_lr(step, TrainCfg, total_steps)

                if args.backend == "torch":
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    for pg in optimizer.param_groups: pg['lr'] = lr
                    
                    logits, loss, _ = model(x, targets=y)
                    raw_loss = loss.item()
                    
                    if not loss_tracker.is_valid(raw_loss):
                        optimizer.zero_grad()
                        continue
                    
                    loss = loss / TrainCfg.grad_accum_steps
                    loss.backward()
                    
                    if (step + 1) % TrainCfg.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    # MLX Path
                    x_mlx = mx.array(x.numpy())
                    y_mlx = mx.array(y.numpy())
                    optimizer.learning_rate = lr
                    
                    loss_mlx = train_step(x_mlx, y_mlx)
                    mx.eval(state)
                    raw_loss = loss_mlx.item()

                loss_tracker.update(raw_loss)
                accum_loss += raw_loss
                
                if step % 10 == 0:
                    dt = time.time() - t0
                    tps = (TrainCfg.batch_size * TrainCfg.seq_len * 10) / dt
                    print(f"Step {step} | Loss: {raw_loss:.4f} | LR: {lr:.2e} | {tps:.0f} tok/s")
                    t0 = time.time()
                
                if step % 1000 == 0:
                    ckpt.save(step, raw_loss)
                
                step += 1
                if step >= total_steps: break

            except Exception as e:
                print(f"💥 Crash: {e}")
                traceback.print_exc()
                ckpt.save(step, 0.0, is_crash=True)
                return

if __name__ == "__main__":
    main()