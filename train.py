import time
import math
import os
import argparse
import signal
import sys
import traceback
import glob
import numpy as np
from dataset import PackedBinDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import deepspeed
from params import Config, TrainCfg
from mlx.utils import tree_map

# --- Argument Parsing (Before backend selection) ---
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--train_dir", type=str, default="data/shards/phase1/train")
parser.add_argument("--val_dir", type=str, default="data/shards/phase1/val")
parser.add_argument(
    "--backend",
    type=str,
    default="mlx" if sys.platform == "darwin" else "deepspeed",
    choices=["torch", "mlx", "deepspeed"],
    help="Backend selection. Defaults to mlx on macOS, deepspeed on Linux/Windows."
)
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

# Parse known args first to check backend
args, remaining = parser.parse_known_args()

# --- Backend-Specific Imports ---
if args.backend == "mlx":
    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as mopt
    from transformer_mlx import LLM
    mx.set_cache_limit(16 * 1024 * 1024 * 1024)
    print("🍏 Using MLX Backend (Apple Silicon Optimized)")

elif args.backend == "deepspeed":
    import deepspeed
    from transformer import LLM
    # Add DeepSpeed arguments to parser
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    print("🚀 Using DeepSpeed Backend (ZeRO-2 + CPU Offload)")

else:  # vanilla torch
    from transformers import Adafactor
    from transformer import LLM
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("🔥 Using PyTorch Backend")


# --- Scheduler (Used for non-DeepSpeed backends) ---
def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr_start * (step / cfg.warmup_steps)
    
    decay_steps = total_steps - cfg.warmup_steps
    step_in_decay = min(step - cfg.warmup_steps, decay_steps)
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_decay / decay_steps))
    decayed = (cfg.lr_start - cfg.lr_end) * cosine_decay + cfg.lr_end
    return decayed


# --- Checkpoint Manager ---
class CheckpointManager:
    def __init__(self, model, optimizer, save_dir="models", backend="torch"):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.backend = backend
        os.makedirs(save_dir, exist_ok=True)

    def save(self, step, loss, is_crash=False, is_best=False):
        prefix = "crash_" if is_crash else "ckpt_"
        if is_best:
            prefix = "best_"
        tag = f"{prefix}step_{step}"

        if self.backend == "deepspeed":
            # DeepSpeed has its own checkpoint format
            self.model.save_checkpoint(self.save_dir, tag=tag)
        elif self.backend == "torch":
            path = os.path.join(self.save_dir, f"{tag}.pt")
            state = {
                "step": step,
                "loss": loss,
                "config": Config.__dict__,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict()
            }
            torch.save(state, path)
        else:  # mlx
            self.model.save_weights(os.path.join(self.save_dir, f"{tag}.safetensors"))

        print(f"💾 Saved: {tag}")
        if not is_crash:
            self._prune()

    def load(self, path):
        print(f"Resuming from {path}...")
        if self.backend == "deepspeed":
            # DeepSpeed loads from directory
            _, client_state = self.model.load_checkpoint(path)
            return client_state.get("step", 0) if client_state else 0
        elif self.backend == "torch":
            c = torch.load(path, map_location="cpu")
            self.model.load_state_dict(c["model"])
            self.optimizer.load_state_dict(c["optim"])
            return c["step"]
        else:  # mlx
            self.model.load_weights(path)
            return 0

    def _prune(self):
        if self.backend == "deepspeed":
            # DeepSpeed checkpoints are directories
            dirs = sorted(
                [d for d in glob.glob(os.path.join(self.save_dir, "ckpt_step_*")) if os.path.isdir(d)],
                key=os.path.getmtime
            )
            for d in dirs[:-3]:
                import shutil
                shutil.rmtree(d)
        else:
            pattern = "*.pt" if self.backend == "torch" else "*.safetensors"
            files = sorted(
                glob.glob(os.path.join(self.save_dir, f"ckpt_step_*{pattern}")),
                key=os.path.getmtime
            )
            for f in files[:-3]:
                os.remove(f)


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
        if not self.history:
            return float('inf')
        return sum(self.history) / len(self.history)

    def is_spike(self, loss_val):
        if len(self.history) < 50:
            return False
        return loss_val > self.spike_threshold * self.running_avg

    def is_valid(self, loss_val):
        return math.isfinite(loss_val)


@torch.no_grad()
def validate(model, val_loader, backend="torch"):
    if backend in ["torch", "deepspeed"]:
        model.eval()
    
    total_loss = 0
    steps = 0
    max_val_steps = 100

    for x, y in val_loader:
        if backend == "deepspeed":
            x, y = x.to(model.device), y.to(model.device)
            _, loss, _ = model(x, targets=y)
        elif backend == "torch":
            x, y = x.cuda(), y.cuda()
            _, loss, _ = model(x, targets=y)
        else:  # mlx
            x_mlx = mx.array(x.numpy())
            y_mlx = mx.array(y.numpy())
            logits, _ = model(x_mlx)
            loss = mx.mean(mnn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                y_mlx.reshape(-1)
            ))
            mx.eval(loss)

        total_loss += float(loss.item())
        steps += 1
        if steps >= max_val_steps:
            break

    avg_loss = total_loss / steps
    perplexity = math.exp(avg_loss)
    
    if backend in ["torch", "deepspeed"]:
        model.train()
    
    return avg_loss, perplexity


def main():
    # === 1. Model & Optimizer Initialization ===
    
    if args.backend == "deepspeed":
        # DeepSpeed: Model in FP32, DS handles mixed precision
        model = LLM(Config)
        
        # DeepSpeed initialize (creates optimizer, scheduler, wraps model)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config={
                "train_batch_size": TrainCfg.batch_size * TrainCfg.grad_accum_steps,
                "train_micro_batch_size_per_gpu": TrainCfg.batch_size,
                "gradient_accumulation_steps": TrainCfg.grad_accum_steps,
                "gradient_clipping": 1.0,
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": TrainCfg.lr_start,
                        "betas": [0.9, 0.95],
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    }
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": TrainCfg.lr_end,
                        "warmup_max_lr": TrainCfg.lr_start,
                        "warmup_num_steps": TrainCfg.warmup_steps,
                        "total_num_steps": 200000
                    }
                }
            }
        )
        
    elif args.backend == "torch":
        from transformers import Adafactor
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
        lr_scheduler = None
        
    else:  # mlx
        model = LLM(Config)
        mx.eval(model.parameters())
        optimizer = mopt.Adafactor(
            learning_rate=TrainCfg.lr_start,
            eps=(1e-30, 1e-3),
        )
        lr_scheduler = None

    # === 2. Checkpoint Manager ===
    ckpt = CheckpointManager(model, optimizer if args.backend != "deepspeed" else None, backend=args.backend)
    start_step = 0
    if args.resume:
        start_step = ckpt.load(args.resume)

    # === 3. Data Loaders ===
    print(f"📂 Loading Binary Dataset from {args.train_dir}...")
    
    train_ds = PackedBinDataset(
        args.train_dir,
        split="train",
        seq_len=TrainCfg.seq_len,
        dtype=np.uint16,
        pattern="*train-*.bin",
    )
    val_ds = PackedBinDataset(
        args.val_dir,
        split="val",
        seq_len=TrainCfg.seq_len,
        pattern="*val-*.bin"
    )

    # Batch size: DeepSpeed controls this internally
    if args.backend == "deepspeed":
        batch_size = model.train_micro_batch_size_per_gpu()
    else:
        batch_size = TrainCfg.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0 if args.backend == "mlx" else 3, # Using PCIe4 instead of 5, so using 3 instead of 4 workers
        pin_memory=(args.backend != "mlx"),
        persistent_workers=(args.backend not in ["mlx", "deepspeed"]),
        prefetch_factor=4 if args.backend != "mlx" else None, # Would be 2 if using PCIe5 (using 4 for PCIe4)
        shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    total_steps = max(
        int(len(train_ds) / (TrainCfg.batch_size * TrainCfg.grad_accum_steps)),
        200_000
    )
    loss_tracker = LossTracker()

    # === 4. Signal Handler ===
    step = start_step
    
    def signal_handler(sig, frame):
        print("\n🛑 Saving & Exiting...")
        ckpt.save(step, 0.0, is_crash=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # === 5. MLX Training Step (if needed) ===
    if args.backend == "mlx":
        def loss_fn(model, x, y):
            logits, _ = model(x)
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            loss = mx.mean(mnn.losses.cross_entropy(logits.astype(mx.float32), y))
            return loss

        loss_and_grad_fn = mnn.value_and_grad(model, loss_fn)
        GRAD_CLIP = 0.5
        accumulated_grads = None

        def train_step_mlx(x, y, accum_step):
            nonlocal accumulated_grads
            loss, grads = loss_and_grad_fn(model, x, y)
            # Scale gradients for accumulation
            grads = tree_map(lambda g: g / TrainCfg.grad_accum_steps, grads)
            
            # Accumulate
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g, accumulated_grads, grads
                )
            
            # Only update on final accumulation step
            if (accum_step + 1) % TrainCfg.grad_accum_steps == 0:
                clipped, grad_norm = mopt.clip_grad_norm(accumulated_grads, max_norm=GRAD_CLIP)
                if math.isfinite(grad_norm.item()):
                    optimizer.update(model, clipped)
                accumulated_grads = None
                return loss, grad_norm
            return loss, mx.array(0.0)

    # === 6. Training Loop ===
    best_val_loss = float('inf')
    accum_loss = 0.0
    t0 = time.time()

    print(f"🔥 Training Start: {total_steps} steps total")

    while step < total_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            # Skip to resume point
            if step < start_step:
                if batch_idx % 100 == 0:
                    print(f"⏩ Skipping to catch up... ({step}/{start_step})", end="\r")
                step += 1
                continue

            try:
                # --- DeepSpeed Path ---
                if args.backend == "deepspeed":
                    x = x.to(model.device)
                    y = y.to(model.device)

                    _, loss, _ = model(x, targets=y)
                    raw_loss = loss.item()

                    if not loss_tracker.is_valid(raw_loss):
                        continue

                    # DeepSpeed handles backward + optimizer step + gradient accumulation
                    model.backward(loss)
                    model.step()

                # --- Vanilla PyTorch Path ---
                elif args.backend == "torch":
                    lr = get_lr(step, TrainCfg, total_steps)
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                    _, loss, _ = model(x, targets=y)
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

                # --- MLX Path ---
                else:
                    lr = get_lr(step, TrainCfg, total_steps)
                    x_mlx = mx.array(x.numpy().astype("int32"))
                    y_mlx = mx.array(y.numpy().astype("int32"))
                    optimizer.learning_rate = lr

                    loss_mlx, grad_norm = train_step_mlx(x_mlx, y_mlx, step)
                    mx.eval(model.parameters(), optimizer.state, loss_mlx)
                    raw_loss = float(loss_mlx.item())

                    if not math.isfinite(raw_loss):
                        print(f"⚠️ NaN loss at step {step}, skipping")
                        continue

                # --- Logging & Checkpointing (All backends) ---
                loss_tracker.update(raw_loss)
                accum_loss += raw_loss

                if step % 25 == 0:
                    dt = time.time() - t0
                    tps = (TrainCfg.batch_size * TrainCfg.seq_len * 25) / dt
                    
                    if args.backend == "deepspeed":
                        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else TrainCfg.lr_start
                    else:
                        current_lr = lr if args.backend != "deepspeed" else 0
                    
                    print(f"Step {step} | Loss: {raw_loss:.4f} | LR: {current_lr:.2e} | {tps:.0f} tok/s")
                    t0 = time.time()

                if step % 1000 == 0 and step > 0:
                    val_loss, val_ppl = validate(model, val_loader, args.backend)
                    print(f"📉 [VAL] Step {step} | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")

                    ckpt.save(step, val_loss, is_best=False)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt.save(step, val_loss, is_best=True)
                        print(f"🌟 New best model found at step {step}!")

                step += 1
                if step >= total_steps:
                    break

            except Exception as e:
                print(f"💥 Crash: {e}")
                traceback.print_exc()
                ckpt.save(step, 0.0, is_crash=True)
                return

    print("✅ Training complete!")
    ckpt.save(step, best_val_loss, is_best=False)


if __name__ == "__main__":
    main()