import argparse
import math
import os
import random
import signal
import sys
import threading
import time
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import Adafactor
from typing import Optional

from transformer import LLM
from dataset import PackedBinDataset, get_dataloader
from utils import *
from params import Config, TrainCfg


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
        dist.barrier()

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
    # torch.cuda.set_per_process_memory_fraction(0.875, device=0) # Fits in 14GB VRAM (Increase this if you have more VRAM available)

    seed_everything(args.seed)

    # Set compile flag BEFORE model creation
    if args.compile:
        Config.compile_layers = True

    # === Model ===
    model = LLM(Config).to(device)

    # === Optimizer / Scheduler ===
    optimizer = Adafactor(
        model.parameters(),
        lr=float(TrainCfg.lr_start),
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=0.9,  # set to None if you want to trade compute for lower VRAM
        weight_decay=0.01,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    scheduler = make_scheduler(
        optimizer,
        total_opt_steps=args.total_opt_steps,
        warmup_steps=TrainCfg.warmup_steps,
        schedule=args.lr_schedule,
    )

    ckpt = CheckpointManager(save_dir=args.save_dir)

    # === State (single load path) ===
    opt_step = 0
    micro_step = 0
    epoch = 0
    micro_step_in_epoch = 0
    best_val_loss = float("inf")

    state = ckpt.load(
        args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if state is not None:
        opt_step = int(state.get("opt_step", 0))
        micro_step = int(state.get("micro_step", 0))
        epoch = int(state.get("epoch", 0))
        micro_step_in_epoch = int(state.get("micro_step_in_epoch", 0))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        set_rng_state(state.get("rng", None))

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
            worker_init_fn=worker_init_fn
        )
        val_sampler = DistributedSampler(
            val_ds,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
            worker_init_fn=worker_init_fn
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

    # === Initialize DataLoaders WITH OFFSET ===
    train_loader = get_dataloader(
        train_ds,
        train_sampler,
        TrainCfg.batch_size,
        args.num_workers,
        args.prefetch_factor,
        shuffle,
        args.seed,
        offset=micro_step_in_epoch,
        drop_last=True,
    )
    val_loader = get_dataloader(
        val_ds,
        val_sampler,
        TrainCfg.batch_size,
        max(0, args.num_workers // 2),
        args.prefetch_factor,
        False,
        args.seed,
        offset=0,
        drop_last=False,
    )

    # Compilation now handled inside LLM.__init__ via Config.compile_layers

    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())

    if is_main_process():
        print(
            f"🔥 Start/Resume: opt_step={opt_step} micro_step={micro_step} "
            f"epoch={epoch} micro_in_epoch={micro_step_in_epoch}"
        )

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_iter = iter(train_loader)
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
            tag=f"interrupt_step_{opt_step}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            client_state=client_state,
            is_crash=True,
        )

    exiting = threading.Event()
    def handle_sigint(sig, frame):
        # Get the current process ID
        import os
        main_pid = os.getpgrp() 
        
        # Only the main process should save checkpoints
        # Workers should just exit
        if is_main_process() and torch.cuda.is_initialized():
            # Prevent multiple triggers if you mash Ctrl+C
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\nSIGINT: saving interrupt checkpoint and exiting...")
            try:
                save_crash_and_exit()
            except Exception as e:
                print(f"Failed to save interrupt checkpoint: {e}")
        
        # Kill the process group to ensure workers die too
        if is_distributed():
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

            x = x.to(device, dtype=torch.long, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

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
                print(f"Crash: {e}")
                traceback.print_exc()
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