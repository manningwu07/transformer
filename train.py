import argparse
import glob
import math
import os
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PackedBinDataset
from params import Config, TrainCfg
from transformer import LLM


def is_distributed() -> bool:
    return int(os.getenv("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.getenv("RANK", "0"))


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def is_main_process() -> bool:
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
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def write_latest(save_dir: str, tag: str):
    latest_path = os.path.join(save_dir, "latest")
    with open(latest_path, "w") as f:
        f.write(tag)


def read_latest(save_dir: str) -> str | None:
    latest_path = os.path.join(save_dir, "latest")
    if os.path.isfile(latest_path):
        with open(latest_path, "r") as f:
            return f.read().strip()
    return None


class CheckpointManager:
    def __init__(self, save_dir: str = "models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        tag: str,
        model,
        optimizer,
        scheduler,
        client_state: dict,
        is_crash: bool = False,
        keep: int = 3,
    ):
        if not is_main_process():
            return

        path = os.path.join(self.save_dir, f"{tag}.pt")
        payload = {
            "model": model.state_dict()
            if not isinstance(model, DDP)
            else model.module.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "client_state": client_state,
        }
        torch.save(payload, path)
        write_latest(self.save_dir, tag)
        print(f"💾 Saved checkpoint: {path}")

        if not is_crash:
            self.prune(keep=keep)

    def load(self, resume: str | None, model, optimizer, scheduler):
        # resume:
        #   None -> load save_dir/latest if exists
        #   "models" -> load models/latest
        #   "models/ckpt_step_123.pt" or ".../ckpt_step_123" -> load that
        if resume is None:
            root = self.save_dir
            tag = read_latest(root)
            if tag is None:
                return None
            path = os.path.join(root, f"{tag}.pt")
        else:
            resume = os.path.abspath(os.path.expanduser(resume))
            if os.path.isdir(resume):
                root = resume
                tag = read_latest(root)
                if tag is None:
                    return None
                path = os.path.join(root, f"{tag}.pt")
            else:
                path = resume
                if not path.endswith(".pt"):
                    path = path + ".pt"
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"--resume path not found: {path}")

        if is_main_process():
            print(f"🔁 Resuming from {path}")

        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["model"]
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        return ckpt.get("client_state", None)

    def prune(self, keep: int = 3):
        if not is_main_process():
            return
        ckpts = sorted(
            glob.glob(os.path.join(self.save_dir, "*.pt")),
            key=os.path.getmtime,
        )
        to_delete = ckpts[:-keep]
        for p in to_delete:
            try:
                os.remove(p)
            except Exception as e:
                print(f"⚠️ Failed pruning {p}: {e}")


@torch.no_grad()
def validate(model, device, val_loader, max_val_steps: int = 100):
    model.eval()
    total_loss = 0.0
    steps = 0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/shards/phase1/train")
    parser.add_argument("--val_dir", type=str, default="data/shards/phase1/val")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--seed", type=int, default=getattr(Config, "seed", 1337))

    parser.add_argument("--total_opt_steps", type=int, default=200_000)
    parser.add_argument("--log_every_opt", type=int, default=25)
    parser.add_argument("--val_every_opt", type=int, default=1000)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    if is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_local_rank())

    device = torch.device("cuda", get_local_rank())
    
    # === Hardware Optimizations ===
    # Enable TF32 for faster matmuls (Tensor Cores)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Allow cuDNN to find the best kernel for your specific hardware
    torch.backends.cudnn.benchmark = True

    # SDPA knobs (FlashAttention / mem-efficient if available)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    seed_everything(args.seed)

    # === Dataset ===
    if is_main_process():
        print(f"📂 Loading datasets: train={args.train_dir} val={args.val_dir}")

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
        dtype=np.uint16,
        pattern="*val-*.bin",
    )

    if is_distributed():
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    rank = get_rank()

    def worker_init_fn(worker_id: int):
        base = args.seed + 1000 * rank + worker_id
        np.random.seed(base)
        random.seed(base)

    train_loader = DataLoader(
        train_ds,
        batch_size=TrainCfg.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers, # num_workers=8 (Optimal for most systems)
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None, # prefetch_factor=4 (default)
        drop_last=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=TrainCfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    # === Model ===
    model = LLM(Config).to(device)

    if args.compile:
        # Keep it simple: compile before DDP for single GPU; for DDP this may vary by version.
        model = torch.compile(model, mode="max-autotune")

    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())

    # === Optimizer (fused AdamW if available) ===
    opt_kwargs = dict(
        lr=float(TrainCfg.lr_start),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
        fused_str = "fused=True"
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        fused_str = "fused=unsupported"

    scheduler = make_scheduler(
        optimizer,
        total_opt_steps=args.total_opt_steps,
        warmup_steps=TrainCfg.warmup_steps,
        schedule=args.lr_schedule,
    )

    if is_main_process():
        print(
            f"🧪 Optimizer: AdamW ({fused_str}) | "
            f"Schedule: warmup={TrainCfg.warmup_steps} + {args.lr_schedule}"
        )

    ckpt = CheckpointManager(save_dir=args.save_dir)

    opt_step = 0
    micro_step = 0
    epoch = 0
    micro_step_in_epoch = 0
    best_val_loss = float("inf")

    # Resume
    state = ckpt.load(args.resume, model=model, optimizer=optimizer, scheduler=scheduler)
    if state is not None:
        opt_step = int(state.get("opt_step", 0))
        micro_step = int(state.get("micro_step", 0))
        epoch = int(state.get("epoch", 0))
        micro_step_in_epoch = int(state.get("micro_step_in_epoch", 0))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        set_rng_state(state.get("rng", None))

    if is_main_process():
        print(
            f"🔥 Start/Resume: opt_step={opt_step} micro_step={micro_step} "
            f"epoch={epoch} micro_in_epoch={micro_step_in_epoch}"
        )

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_iter = iter(train_loader)
    if micro_step_in_epoch > 0:
        if is_main_process():
            print(f"⏩ Skipping {micro_step_in_epoch} batches to restore dataloader position...")
        for _ in range(micro_step_in_epoch):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)

    optimizer.zero_grad(set_to_none=True)

    def save_crash_and_exit():
        client_state = {
            "opt_step": int(opt_step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "micro_step_in_epoch": int(micro_step_in_epoch),
            "best_val_loss": float(best_val_loss),
            "rng": get_rng_state(),
        }
        ckpt.save(
            tag=f"crash_step_{opt_step}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            client_state=client_state,
            is_crash=True,
        )

    def handle_sigint(sig, frame):
        if is_main_process():
            print("\n🛑 SIGINT: saving crash checkpoint and exiting...")
        save_crash_and_exit()
        if is_distributed():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.time()
    loss_window = []
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    model.train()
    while opt_step < args.total_opt_steps:
        try:
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                micro_step_in_epoch = 0
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)

            raw_loss = float(loss.item())
            if not math.isfinite(raw_loss):
                micro_step += 1
                micro_step_in_epoch += 1
                continue

            # grad accumulation
            loss = loss / float(TrainCfg.grad_accum_steps)
            loss.backward()

            micro_step += 1
            micro_step_in_epoch += 1

            loss_window.append(raw_loss)
            if len(loss_window) > 100:
                loss_window.pop(0)

            is_boundary = (micro_step % int(TrainCfg.grad_accum_steps)) == 0
            if is_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                opt_step += 1

                if opt_step % args.log_every_opt == 0 and is_main_process():
                    dt = time.time() - t0
                    toks = (
                        TrainCfg.batch_size
                        * TrainCfg.grad_accum_steps
                        * TrainCfg.seq_len
                        * max(1, world_size)
                        * args.log_every_opt
                    )
                    tps = toks / max(dt, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]
                    avg = sum(loss_window) / max(1, len(loss_window))
                    print(
                        f"Opt {opt_step} | Micro {micro_step} | "
                        f"Loss {raw_loss:.4f} (avg {avg:.4f}) | "
                        f"LR {lr:.2e} | {tps:.0f} tok/s"
                    )
                    t0 = time.time()

                if opt_step % args.val_every_opt == 0:
                    val_loss, val_ppl = validate(model, device, val_loader, max_val_steps=100)
                    if is_main_process():
                        print(
                            f"📉 [VAL] Opt {opt_step} | Loss {val_loss:.4f} | PPL {val_ppl:.2f}"
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        client_state = {
                            "opt_step": int(opt_step),
                            "micro_step": int(micro_step),
                            "epoch": int(epoch),
                            "micro_step_in_epoch": int(micro_step_in_epoch),
                            "best_val_loss": float(best_val_loss),
                            "rng": get_rng_state(),
                        }
                        ckpt.save(
                            tag=f"best_step_{opt_step}",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            client_state=client_state,
                            is_crash=False,
                        )

        except Exception as e:
            if is_main_process():
                print(f"💥 Crash: {e}")
                traceback.print_exc()
            save_crash_and_exit()
            if is_distributed():
                dist.destroy_process_group()
            return

    if is_main_process():
        print("✅ Training complete!")

    client_state = {
        "opt_step": int(opt_step),
        "micro_step": int(micro_step),
        "epoch": int(epoch),
        "micro_step_in_epoch": int(micro_step_in_epoch),
        "best_val_loss": float(best_val_loss),
        "rng": get_rng_state(),
    }
    ckpt.save(
        tag=f"ckpt_step_{opt_step}",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        client_state=client_state,
        is_crash=False,
    )

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()