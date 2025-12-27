import argparse
import os
import random
import signal
import sys
import threading
import time
import gc
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import Adafactor

from transformer import LLM
from dataset import PackedBinDataset, get_dataloader
from utils import *
from params import Config, TrainCfg

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/shards/phase1/train")
    parser.add_argument("--val_dir", type=str, default="data/shards/phase1/val")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--cuda_graphs", action="store_true")
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
        Config.compile_mode = "model"
    else:
        Config.compile_mode = "none"

    # === Model ===
    model = LLM(Config).to(device)
    
    # Whole-model compile (AFTER .to(device), BEFORE DDP)
    if getattr(model, "_needs_whole_model_compile", False):
        print("⚡ Compiling entire model (this may take 2-5 minutes on first forward   backward)...")
        
        # Suppress excessive recompilation from dynamic shapes
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.cache_size_limit = 64
        
        model = torch.compile(
            model,
            mode="max-autotune",       # or "reduce-overhead" for CUDA graphs (needs static shapes)
            fullgraph=True,      # Allow graph breaks (safer with checkpointing)
            dynamic=False,        # Static shapes = faster compiled code
        )
        print("✅ Model compiled (warmup will happen on first batch)")

    # === Optimizer / Scheduler ===
    optimizer = Adafactor(
        model.parameters(),
        lr=float(TrainCfg.lr_start),
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,  # set to None if you want to trade compute for lower VRAM
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
            drop_last=True,
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
        drop_last=True,
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
    
    use_cuda_graphs = bool(args.cuda_graphs) and not is_distributed()
    if args.cuda_graphs and is_distributed() and is_main_process():
        print("⚠️  --cuda_graphs disabled under DDP (NCCL not capture-safe).")

    # For CUDA graph replay, grads must exist (cannot be None) across replays.
    # So we must NOT use set_to_none=True once graphs are enabled.
    optimizer.zero_grad(set_to_none=not use_cuda_graphs)

    micro_graph = None
    static_x = None
    static_y = None
    static_loss = None

    if use_cuda_graphs:
        graph_stream = torch.cuda.Stream()
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        
        if is_main_process():
            print("📌 Capturing manual CUDA graph (forward + backward micro-step)...")

        static_x = torch.empty(
            (int(TrainCfg.batch_size), int(TrainCfg.seq_len)),
            device=device,
            dtype=torch.long,
        )
        static_y = torch.empty_like(static_x)
        static_loss = torch.empty((), device=device, dtype=torch.float32)

        # Dummy data just to compile kernels, allocate buffers, and capture.
        static_x.random_(0, int(Config.vocab_size))
        static_y.random_(0, int(Config.vocab_size))

         # Warmup + capture must run on the SAME non-default stream to avoid
        # autograd trying to join legacy stream with capture stream.
        torch.cuda.synchronize()
        with torch.cuda.stream(graph_stream):
            # Wait for any work enqueued on legacy default stream
            graph_stream.wait_stream(torch.cuda.default_stream())

            # Warmup (outside capture): trigger torch.compile/autotune and allocate buffers.
            for _ in range(5):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, warm_loss, _ = model(static_x, targets=static_y)
                (warm_loss / float(TrainCfg.grad_accum_steps)).backward()
                optimizer.zero_grad(set_to_none=False)
                del warm_loss

            # Make sure no Python refs keep autograd graphs alive
            gc.collect()
            torch.cuda.synchronize()
            optimizer.zero_grad(set_to_none=False)
            torch.cuda.synchronize()

        micro_graph = torch.cuda.CUDAGraph()
        
        # Use a dedicated graph memory pool handle to reduce allocator surprises.
        graph_pool = torch.cuda.graphs.graph_pool_handle()

        with torch.cuda.stream(graph_stream):
            graph_stream.wait_stream(torch.cuda.default_stream())
            try:
                with torch.cuda.graph(
                    micro_graph,
                    pool=graph_pool,
                    capture_error_mode="global",
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss, _ = model(static_x, targets=static_y)
                    static_loss.copy_(loss)
                    (loss / float(TrainCfg.grad_accum_steps)).backward()
            except TypeError:
                # Older torch may not support capture_error_mode kwarg.
                with torch.cuda.graph(micro_graph, pool=graph_pool):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss, _ = model(static_x, targets=static_y)
                    static_loss.copy_(loss)
                    (loss / float(TrainCfg.grad_accum_steps)).backward()

        torch.cuda.synchronize()
        if is_main_process():
            print("✅ CUDA graph captured.")
    
    def get_eager_model(m):
        # If using DDP, unwrap.
        if isinstance(m, DDP):
            m = m.module
        # If compiled, prefer the original eager module.
        return getattr(m, "_orig_mod", m)

    eager_model_for_val = get_eager_model(model)

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

    def handle_sigint(sig, frame):        
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
    
    # Compiled autograd context manager
    if use_cuda_graphs:
        compiled_autograd_ctx = None    
    if getattr(Config, "use_compiled_autograd", False):
        try:
            import torch._dynamo.compiled_autograd as ca
            compiled_autograd_ctx = ca.enable(
                compiler=lambda gm: torch.compile(gm, mode="max-autotune", fullgraph=True)
            )
            print("⚡ compiled_autograd enabled (experimental)")
        except Exception as e:
            print(f"⚠️ compiled_autograd not available: {e}")
            compiled_autograd_ctx = None
            
    # --------------------------------------------
    # Compile forward+loss
    # --------------------------------------------
    def maybe_cudagraph_step_begin() -> None:
        # Only for Inductor-managed cudagraphs. When using manual
        # torch.cuda.CUDAGraph, this is unnecessary.
        if not args.compile or use_cuda_graphs:
            return
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass

    def forward_loss(x, y):
        # forward + loss only (safe for torch.compile)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y)
        return loss

    forward_loss_compiled = forward_loss
    if args.compile:
        forward_loss_compiled = torch.compile(
            forward_loss,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
            
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
            
            next_micro_step = int(micro_step) + 1
            will_boundary = (
                next_micro_step > 0
                and next_micro_step % int(TrainCfg.grad_accum_steps) == 0
            )
            
            if use_cuda_graphs:
                assert static_x is not None and static_y is not None
                assert static_loss is not None and micro_graph is not None
                assert "graph_stream" in locals()

                # Copy + replay on the SAME non-default stream used for capture.
                with torch.cuda.stream(graph_stream):
                    static_x.copy_(x, non_blocking=True)
                    static_y.copy_(y, non_blocking=True)
                    micro_graph.replay()

                # Ensure optimizer step on default stream is ordered after replay.
                torch.cuda.default_stream().wait_stream(graph_stream)
                loss_for_log = static_loss
            else:
                x = x.to(device, dtype=torch.long, non_blocking=True).clone()
                y = y.to(device, dtype=torch.long, non_blocking=True).clone()

                maybe_cudagraph_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss, _ = model(x, targets=y)

                loss_for_backward = loss / float(TrainCfg.grad_accum_steps)
                maybe_cudagraph_step_begin()

                if compiled_autograd_ctx is not None:
                    with compiled_autograd_ctx:
                        loss_for_backward.backward()
                else:
                    loss_for_backward.backward()

                loss_for_log = loss
            

            raw_loss = None
            if will_boundary:
                raw_loss = float(loss_for_log.detach().float().cpu().item())

            micro_step += 1
            micro_step_in_epoch += 1

            if will_boundary:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert raw_loss is not None
                loss_window.append(raw_loss)
                if len(loss_window) > 100: loss_window.pop(0)                
    
                if not torch.isfinite(total_norm):
                    print(f"⚠️ Skip step {opt_step}: Gradient norm is {total_norm.item()}")
                    optimizer.zero_grad(set_to_none=not use_cuda_graphs)
                    continue
                
                # optimizer step is outside the compiled train_step
                maybe_cudagraph_step_begin()
                optimizer.step()
                optimizer.zero_grad(set_to_none=not use_cuda_graphs)
                scheduler.step()

                opt_step += 1

                # 3. Log (Only on Boundary)
                if opt_step % args.log_every_opt == 0 and is_main_process():
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
                        f"LR {lr:.2e} | {tps:.0f} tok/s"
                    )
                    
                    torch.cuda.reset_peak_memory_stats()
                    # run one full optimizer step (all micro steps)
                    print("max allocated GB", torch.cuda.max_memory_allocated() / 1e9)
                    print("max reserved  GB", torch.cuda.max_memory_reserved() / 1e9)
                    
                    last_log_t = end_compute_t

                if opt_step % args.val_every_opt == 0:
                    val_loss, val_ppl = validate(eager_model_for_val, device, val_loader)
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