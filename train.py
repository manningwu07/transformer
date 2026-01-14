import argparse
import os
import random
import signal
import time
import traceback

from bench.fused_adafactor import FusedAdafactor
from transformer import _ce_calls

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.inductor_cache")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformer import LLM
from dataset import PackedBinDataset, get_dataloader
from utils import *
from params import Config, TrainCfg
import signal
import atexit

def cleanup_gpu():
    """Force CUDA cleanup on exit"""
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

atexit.register(cleanup_gpu)

@torch.compiler.disable
def safe_step(optimizer, scheduler):
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    
@torch.no_grad()
def _mark_fp8_dirty_after_step(model) -> None:
    """
    After optimizer.step(), mark FP8 buffers as dirty so they get re-quantized
    once per optimizer update (NOT per micro-step).
    Works for eager, compiled, and DDP-wrapped models.
    """
    # unwrap DDP
    if isinstance(model, DDP):
        model = model.module
    # unwrap torch.compile
    model = getattr(model, "_orig_mod", model)

    # walk modules; call mark_step() if present (your fused MLP exposes it)
    for m in model.modules():
        mark = getattr(m, "mark_step", None)
        if callable(mark):
            mark()

def main():
    
    def discover_val_buckets(val_dir: str):
        """
        If val_dir has subfolders (e.g. val/finemath, val/stackedu, val/anchor),
        return those as buckets. Otherwise return {"val": val_dir}.
        """
        if not os.path.isdir(val_dir):
            return {"val": val_dir}

        # If there are .bin files directly inside val_dir, treat as single bucket.
        direct_bins = glob.glob(os.path.join(val_dir, "*.bin"))
        if direct_bins:
            return {"val": val_dir}

        # Otherwise, treat each immediate subdir containing .bin (recursively) as a bucket.
        buckets = {}
        for name in sorted(os.listdir(val_dir)):
            p = os.path.join(val_dir, name)
            if not os.path.isdir(p):
                continue
            has_bins = bool(glob.glob(os.path.join(p, "**", "*.bin"), recursive=True))
            if has_bins:
                buckets[name] = p

        return buckets if buckets else {"val": val_dir}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/shards/phase2-final/train")
    parser.add_argument("--val_dir", type=str, default="data/shards/phase2-final/val")
    parser.add_argument(
        "--val_metric",
        type=str,
        default="avg",
        choices=["avg", "finemath", "stackedu", "anchor", "val"],
        help="Which validation bucket determines best checkpoint.",
    )
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--seed", type=int, default=getattr(Config, "seed", 1337))

    parser.add_argument("--total_opt_steps", type=int, default=10_000)
    parser.add_argument("--log_every_opt", type=int, default=25)
    parser.add_argument("--val_every_opt", type=int, default=100)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
    )
    parser.add_argument("--init_from", type=str, default=None, help="Load model weights only, reset optimizer/scheduler/steps")

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
        Config.compile_mode = "model"
    else:
        Config.compile_mode = "none"

    # === Model ===
    model = LLM(Config).to(device)

    # === Optimizer / Scheduler ===
    optimizer = FusedAdafactor(
        model.parameters(),
        lr=float(TrainCfg.lr_start),
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        weight_decay=0.01
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

    if args.init_from is not None:
        # Load ONLY model weights; do NOT load optimizer/scheduler/client_state.
        _ = ckpt.load(
            args.init_from,
            model=model,
            optimizer=None,
            scheduler=None,
        )
        # Reset run state explicitly.
        opt_step = 0
        micro_step = 0
        epoch = 0
        micro_step_in_epoch = 0
        best_val_loss = float("inf")
        # Fresh RNG for the new phase (recommended)
        seed_everything(args.seed)
        if is_main_process():
            print(f"🧪 init_from={args.init_from} (model-only). Optimizer/scheduler reset.")
    else:
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
    # Whole-model compile (AFTER .to(device), BEFORE DDP)
    if getattr(model, "_needs_whole_model_compile", False):
        print("⚡ Compiling entire model (this may take 2-5 minutes on first forward + backward)...")
        
        # Suppress excessive recompilation from dynamic shapes
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.cache_size_limit = 64
        
        model = torch.compile(
            model,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,      # No graph breaks
            dynamic=False,        # Static shapes = faster compiled code
        )
        # 2. INSERT WARMUP PHASE HERE
        print("🧪 Starting Warmup Phase (Seeding Triton Kernels)...")
        # Use BS=1 to ensure we have maximum VRAM headroom for autotuning
        warmup_bs = 1 
        dummy_x = torch.zeros((warmup_bs, TrainCfg.seq_len), dtype=torch.long, device=device)
        dummy_y = torch.zeros((warmup_bs, TrainCfg.seq_len), dtype=torch.long, device=device)

        # We must run BOTH forward and backward to compile the full autograd graph
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # First pass: Forward + Loss
            _, loss, _ = model(dummy_x, targets=dummy_y)
            # Second pass: Backward (This compiles the gradient kernels)
            loss.backward()
        
        # Cleanup warmup artifacts
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache() 
        torch.cuda.synchronize()
        print("✅ Warmup complete. Kernels cached. Ready for real data.")
        
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
    # ---- Build val loaders per bucket ----
    val_buckets = discover_val_buckets(args.val_dir)
    if is_main_process():
        print(f"🔎 Val buckets: {list(val_buckets.keys())}")

    val_loaders = {}
    for name, vdir in val_buckets.items():
        vds = PackedBinDataset(
            vdir,
            split="val",
            seq_len=TrainCfg.seq_len,
            dtype=np.uint16,
            # More robust: accept any bin names in bucket dirs.
            pattern="*.bin",
        )
        vloader = get_dataloader(
            vds,
            sampler=None,
            batch_size=TrainCfg.batch_size,
            num_workers=max(0, args.num_workers // 2),
            prefetch_factor=args.prefetch_factor,
            shuffle=False,
            seed=args.seed,
            offset=0,
            drop_last=True,
        )
        val_loaders[name] = vloader

    if is_distributed():
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        shuffle = False
    else:
        train_sampler = None
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
    
    def get_eager_model(m):
        # If using DDP, unwrap.
        if isinstance(m, DDP):
            m = m.module
        # If compiled, prefer the original eager module.
        return getattr(m, "_orig_mod", m)

    eager_model_for_val = get_eager_model(model)

    def save_current_state(tag, is_crash=False):
        """Helper to centralize saving logic"""
        if not is_main_process(): return
        
        client_state = {
            "opt_step": int(opt_step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "micro_step_in_epoch": int(micro_step_in_epoch),
            "best_val_loss": float(best_val_loss),
            "rng": get_rng_state(),
        }
        # We unwrap DDP and Compile before saving
        m_to_save = model.module if isinstance(model, DDP) else model
        if hasattr(m_to_save, "_orig_mod"):
            m_to_save = m_to_save._orig_mod
            
        ckpt.save(
            tag=tag,
            model=m_to_save,
            optimizer=optimizer,
            scheduler=scheduler,
            client_state=client_state,
            is_crash=is_crash,
        )

    
    def handle_sigint(sig, frame):
        # 1. Immediately ignore further SIGINTs so we don't loop
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        if is_main_process():
            # 2. Disable watchdog so it doesn't kill us mid-save
            signal.alarm(0)
            print(f"\nExiting... Saving step {opt_step}. STAND BY.")
            try:
                # Force a synchronous save
                save_current_state(f"interrupt_step_{opt_step}", is_crash=True)
                print("✅ Save successful.")
            except Exception as e:
                print(f"❌ Save failed: {e}")
            finally:
                cleanup_gpu()
                # 3. Use os._exit to bypass any other try/finally blocks that might hang
                os._exit(0) 
        else:
            # Ranks > 0 just exit quietly
            os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    
    # Compiled autograd context manager
    compiled_autograd_ctx = None
    if getattr(Config, "use_compiled_autograd", False):
        try:
            import torch._dynamo.compiled_autograd as ca
            compiled_autograd_ctx = ca.enable(
                compiler=lambda gm: torch.compile(gm, mode="max-autotune-no-cudagraphs", fullgraph=True)
            )
            print("⚡ compiled_autograd enabled (experimental)")
        except Exception as e:
            print(f"⚠️ compiled_autograd not available: {e}")
            compiled_autograd_ctx = None
            
    # === Training Loop ===
    t0 = time.time()
    last_log_t = t0
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
            
            # For reduce-overhead / cudagraph replay, mark step begin before
            # EACH compiled invocation. Safe to call even if not using cudagraphs.
            if args.compile:
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)


            # Keep a reference only within this iteration.
            loss_for_backward = loss / float(TrainCfg.grad_accum_steps)
            if compiled_autograd_ctx is not None:
                with compiled_autograd_ctx:
                    loss_for_backward.backward()
            else:
                loss_for_backward.backward()
                
            micro_step += 1
            micro_step_in_epoch += 1

            is_boundary = (micro_step > 0 and micro_step % int(TrainCfg.grad_accum_steps) == 0)
            if is_boundary:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss_window.append(loss.detach())  # tensor on GPU
                total_norm_t = total_norm.detach()
                if len(loss_window) > 100: loss_window.pop(0)                
    
                if torch.isfinite(total_norm):
                    safe_step(optimizer, scheduler)
                    _mark_fp8_dirty_after_step(model)
                else:
                    print(f"⚠️ Skip step {opt_step}: Gradient norm is {total_norm.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                opt_step += 1

                # 3. Log (Only on Boundary)
                if opt_step % args.log_every_opt == 0 and is_main_process():
                    print("CE calls:", dict(_ce_calls))
                    assert _ce_calls["torch"] == 0, "Fell back to torch CE unexpectedly"
                    # Sync ONCE for logging (this will pull all needed scalars)
                    raw_loss = float(loss.detach().float().item())
                    norm_val = float(total_norm_t.float().item())

                    # If you want avg loss, compute it on GPU then item() once:
                    lw = torch.stack([t.float() for t in loss_window[-100:]])
                    avg = float(lw.mean().item())

                    end_compute_t = time.time()
                    dt = end_compute_t - last_log_t
                    
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
                        f"LR {lr:.2e} | Norm {norm_val:.4f} | {tps:.0f} tok/s"
                    )
                    
                    # run one full optimizer step (all micro steps)
                    print("max allocated GB", torch.cuda.max_memory_allocated() / 1e9)
                    print("max reserved  GB", torch.cuda.max_memory_reserved() / 1e9)
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                    last_log_t = end_compute_t

                if opt_step % args.val_every_opt == 0:
                    results = validate_multi(
                        eager_model_for_val,
                        device,
                        val_loaders,
                        max_val_steps=100,
                    )

                    # Print each bucket, plus an average.
                    if is_main_process():
                        keys = sorted(results.keys())
                        avg_loss = sum(results[k][0] for k in keys) / max(1, len(keys))
                        avg_ppl = math.exp(avg_loss)
                        parts = []
                        for k in keys:
                            loss_k, ppl_k = results[k]
                            parts.append(f"{k}: {loss_k:.4f} (ppl {ppl_k:.2f})")
                        print(
                            f"📉 [VAL] Opt {opt_step} | "
                            + " | ".join(parts)
                            + f" | avg: {avg_loss:.4f} (ppl {avg_ppl:.2f})"
                        )

                    save_current_state(f"ckpt_step_{opt_step}", is_crash=False)

                    # Choose which bucket controls best checkpoint.
                    metric = args.val_metric
                    if metric == "avg":
                        score = (
                            sum(v[0] for v in results.values()) / max(1, len(results))
                        )
                    else:
                        # if metric bucket missing, fall back to avg
                        score = results.get(metric, (float("inf"), 0.0))[0]
                        if not math.isfinite(score):
                            score = (
                                sum(v[0] for v in results.values())
                                / max(1, len(results))
                            )

                    if score < best_val_loss:
                        best_val_loss = float(score)
                        save_current_state(f"best_step_{opt_step}", is_crash=False)
                    last_log_t = time.time()

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