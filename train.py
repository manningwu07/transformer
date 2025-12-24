import argparse
import glob
import math
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import asdict

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PackedBinDataset
from params import Config, TrainCfg
from transformer import LLM


def is_main_process(engine) -> bool:
    return getattr(engine, "global_rank", 0) == 0


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
    def __init__(self, engine, save_dir: str = "models"):
        self.engine = engine
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        opt_step: int,
        micro_step: int,
        epoch: int,
        micro_step_in_epoch: int,
        val_loss: float,
        is_crash: bool = False,
        is_best: bool = False,
    ):
        if not is_main_process(self.engine):
            return

        prefix = "crash_" if is_crash else "ckpt_"
        if is_best:
            prefix = "best_"
        tag = f"{prefix}step_{opt_step}"

        client_state = {
            "opt_step": int(opt_step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "micro_step_in_epoch": int(micro_step_in_epoch),
            "val_loss": float(val_loss),
            "rng": get_rng_state(),
            "config": asdict(Config) if hasattr(Config, "__dataclass_fields__") else {},
            "train_cfg": asdict(TrainCfg) if hasattr(TrainCfg, "__dataclass_fields__") else {},
        }

        self.engine.save_checkpoint(self.save_dir, tag=tag, client_state=client_state)
        write_latest(self.save_dir, tag)
        print(f"💾 Saved checkpoint: {os.path.join(self.save_dir, tag)}")

        if not is_crash:
            self.prune(keep=3)

    def load(self, resume: str | None):
        # resume can be:
        #   None -> load save_dir/latest if exists
        #   "models" -> load models/latest
        #   "models/ckpt_step_123" -> load that exact tag
        if resume is None:
            root = self.save_dir
            tag = read_latest(root)
            if tag is None:
                return None
        else:
            resume = os.path.abspath(os.path.expanduser(resume))
            if os.path.isdir(resume):
                # if resume dir contains "latest", treat as root
                if os.path.isfile(os.path.join(resume, "latest")):
                    root = resume
                    tag = read_latest(root)
                    if tag is None:
                        return None
                else:
                    # assume it's a tag dir
                    root = os.path.dirname(resume)
                    tag = os.path.basename(resume)
            else:
                raise FileNotFoundError(f"--resume path not found: {resume}")

        if is_main_process(self.engine):
            print(f"🔁 Resuming from {root} tag={tag}")

        _, client_state = self.engine.load_checkpoint(root, tag=tag)
        return client_state

    def prune(self, keep: int = 3):
        if not is_main_process(self.engine):
            return

        ckpt_dirs = sorted(
            [
                d
                for d in glob.glob(os.path.join(self.save_dir, "ckpt_step_*"))
                if os.path.isdir(d)
            ],
            key=os.path.getmtime,
        )
        to_delete = ckpt_dirs[:-keep]
        for d in to_delete:
            try:
                import shutil

                shutil.rmtree(d)
            except Exception as e:
                print(f"⚠️ Failed pruning {d}: {e}")


@torch.no_grad()
def validate(engine, val_loader, max_val_steps: int = 100):
    engine.eval()

    total_loss = 0.0
    steps = 0

    for x, y in val_loader:
        x = x.to(engine.device, non_blocking=True)
        y = y.to(engine.device, non_blocking=True)
        _, loss, _ = engine(x, targets=y)
        total_loss += float(loss.item())
        steps += 1
        if steps >= max_val_steps:
            break

    avg_loss = total_loss / max(1, steps)
    ppl = math.exp(avg_loss)
    engine.train()
    return avg_loss, ppl


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "-1")))

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This is DeepSpeed CUDA-only.")

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    # Encourage Flash/mem-efficient SDPA for long context later
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    seed_everything(args.seed)

    # === Dataset ===
    if args.local_rank in [-1, 0]:
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

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, seed=args.seed, drop_last=True
        )
        val_sampler = DistributedSampler(val_ds, shuffle=False, seed=args.seed, drop_last=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    def worker_init_fn(worker_id: int):
        # Make worker RNG deterministic too
        base = args.seed + 1000 * rank + worker_id
        np.random.seed(base)
        random.seed(base)

    train_loader = DataLoader(
        train_ds,
        batch_size=TrainCfg.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
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

    # === Model + DeepSpeed ===
    model = LLM(Config)

    ds_config = {
        "train_batch_size": TrainCfg.batch_size * TrainCfg.grad_accum_steps * max(world_size, 1),
        "train_micro_batch_size_per_gpu": TrainCfg.batch_size,
        "gradient_accumulation_steps": TrainCfg.grad_accum_steps,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": TrainCfg.lr_start,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": TrainCfg.lr_end,
                "warmup_max_lr": TrainCfg.lr_start,
                "warmup_num_steps": TrainCfg.warmup_steps,
                "total_num_steps": args.total_opt_steps,
            },
        },
    }

    engine, _, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        args=args,
    )

    ckpt = CheckpointManager(engine, save_dir=args.save_dir)
    state = ckpt.load(args.resume)

    opt_step = 0
    micro_step = 0
    epoch = 0
    micro_step_in_epoch = 0
    best_val_loss = float("inf")

    if state is not None:
        opt_step = int(state.get("opt_step", 0))
        micro_step = int(state.get("micro_step", 0))
        epoch = int(state.get("epoch", 0))
        micro_step_in_epoch = int(state.get("micro_step_in_epoch", 0))
        best_val_loss = float(state.get("val_loss", best_val_loss))
        set_rng_state(state.get("rng", None))

    if is_main_process(engine):
        print(
            f"🔥 Start/Resume: opt_step={opt_step} micro_step={micro_step} "
            f"epoch={epoch} micro_in_epoch={micro_step_in_epoch}"
        )

    # Set sampler epoch and skip within-epoch batches to align dataloader position
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_iter = iter(train_loader)
    if micro_step_in_epoch > 0:
        if is_main_process(engine):
            print(f"⏩ Skipping {micro_step_in_epoch} batches to restore dataloader position...")
        for _ in range(micro_step_in_epoch):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)

    def handle_sigint(sig, frame):
        if is_main_process(engine):
            print("\n🛑 SIGINT: saving crash checkpoint and exiting...")
        ckpt.save(
            opt_step=opt_step,
            micro_step=micro_step,
            epoch=epoch,
            micro_step_in_epoch=micro_step_in_epoch,
            val_loss=0.0,
            is_crash=True,
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.time()
    loss_window = []

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

            x = x.to(engine.device, non_blocking=True)
            y = y.to(engine.device, non_blocking=True)

            _, loss, _ = engine(x, targets=y)
            raw_loss = float(loss.item())
            if not math.isfinite(raw_loss):
                micro_step += 1
                micro_step_in_epoch += 1
                continue

            engine.backward(loss)
            engine.step()

            micro_step += 1
            micro_step_in_epoch += 1

            loss_window.append(raw_loss)
            if len(loss_window) > 100:
                loss_window.pop(0)

            if engine.is_gradient_accumulation_boundary():
                opt_step += 1

                if opt_step % args.log_every_opt == 0 and is_main_process(engine):
                    dt = time.time() - t0
                    toks = (
                        TrainCfg.batch_size
                        * TrainCfg.grad_accum_steps
                        * TrainCfg.seq_len
                        * max(1, world_size)
                        * args.log_every_opt
                    )
                    tps = toks / max(dt, 1e-6)
                    lr = engine.get_lr()[0] if hasattr(engine, "get_lr") else None
                    lr_str = f"{lr:.2e}" if lr is not None else "n/a"
                    avg = sum(loss_window) / max(1, len(loss_window))
                    print(
                        f"Opt {opt_step} | Micro {micro_step} | "
                        f"Loss {raw_loss:.4f} (avg {avg:.4f}) | "
                        f"LR {lr_str} | {tps:.0f} tok/s"
                    )
                    t0 = time.time()

                if opt_step % args.val_every_opt == 0:
                    val_loss, val_ppl = validate(engine, val_loader, max_val_steps=100)
                    if is_main_process(engine):
                        print(
                            f"📉 [VAL] Opt {opt_step} | "
                            f"Loss {val_loss:.4f} | PPL {val_ppl:.2f}"
                        )

                    ckpt.save(
                        opt_step=opt_step,
                        micro_step=micro_step,
                        epoch=epoch,
                        micro_step_in_epoch=micro_step_in_epoch,
                        val_loss=val_loss,
                        is_crash=False,
                        is_best=False,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt.save(
                            opt_step=opt_step,
                            micro_step=micro_step,
                            epoch=epoch,
                            micro_step_in_epoch=micro_step_in_epoch,
                            val_loss=val_loss,
                            is_crash=False,
                            is_best=True,
                        )

        except Exception as e:
            if is_main_process(engine):
                print(f"💥 Crash: {e}")
                traceback.print_exc()
            ckpt.save(
                opt_step=opt_step,
                micro_step=micro_step,
                epoch=epoch,
                micro_step_in_epoch=micro_step_in_epoch,
                val_loss=0.0,
                is_crash=True,
            )
            return

    if is_main_process(engine):
        print("✅ Training complete!")

    ckpt.save(
        opt_step=opt_step,
        micro_step=micro_step,
        epoch=epoch,
        micro_step_in_epoch=micro_step_in_epoch,
        val_loss=best_val_loss,
        is_crash=False,
        is_best=False,
    )


if __name__ == "__main__":
    main()