# 🧠 LLM Pipeline: RTX 5080 Training (1B MLA Model)

This repository contains a high-performance LLM training pipeline optimized for **NVIDIA RTX 50-series (Blackwell)** hardware and **Apple Silicon (M4 Pro)** via MLX. 

The architecture features **Multi-Head Latent Attention (MLA)** with Decoupled RoPE, enabling massive context windows with minimal KV-cache memory overhead.

## 🛠 RTX 5080 Environment Setup

Follow these steps to configure your Linux environment.

### 1. Conda Environment Initial Setup
We use `conda-forge` for system dependencies and `pip` for specialized deep learning libraries like DeepSpeed.

```bash
# Create/Activate environment
conda activate transformer-lab

# Update existing environment to latest stable versions
conda update -n transformer-lab --all -y

# Install build dependencies (required for DeepSpeed fused kernels)
conda install -n transformer-lab -c conda-forge ninja py-boost -y

# Install DeepSpeed & Transformers
pip3 install deepspeed transformers
```

### 2. File Structure
Ensure your data is organized as follows. The `PackedBinDataset` expects raw `.bin` shards.

```text
.
├── train.py            # Unified training script (DeepSpeed/Torch/MLX)
├── params.py           # Model & Training configs (1B / 16k context)
├── transformer.py      # PyTorch MLA implementation
├── transformer_mlx.py  # Apple Silicon MLA implementation
├── dataset.py          # Fast Memmap Binary Loader
├── ds_config.json      # DeepSpeed ZeRO-2 Configuration
└── data/
    └── shards/
        └── phase1/
            ├── train/  # e.g., phase1-train-001.bin
            └── val/    # e.g., phase1-val-001.bin
```

---

## 🚀 Training Execution

### Optimization Profile: RTX 5080 (16GB VRAM)
We use **DeepSpeed ZeRO Stage 2 + CPU Offload**. 
- **ZeRO-2:** Partitions gradients and optimizer states.
- **CPU Offload:** Moves the 8GB+ Adam optimizer states (Momentum/Variance) to your 24GB System RAM, freeing up VRAM for 16k+ context lengths.

#### Run Training:
```bash
deepspeed --num_gpus=1 train.py --backend deepspeed
```

#### Run with Resume:
```bash
deepspeed --num_gpus=1 train.py --backend deepspeed --resume models/ckpt_step_5000
```

---

## ⚙️ Configuration (`params.py`)

To switch between standard pretraining and long-context fine-tuning, edit the `MODE` in `params.py`:

- `pretrain_5080`: 2048 Context, Batch Size 12 (Higher throughput).
- `longctx_5090`: 16384 Context (Leverages MLA compression).

---

## 🛠 Troubleshooting

### 1. Fused Kernels Fail to Compile
If DeepSpeed fails to build its C++ extensions, ensure `ninja` is visible:
```bash
which ninja
# If not found, run: conda install -c conda-forge ninja
```

### 2. Out of Memory (OOM)
If you hit OOM on your RTX 5080:
1. Decrease `batch_size` in `params.py`.
2. Increase `grad_accum_steps` to maintain the effective global batch size.
3. Ensure `offload_optimizer` is set to `cpu` in `ds_config.json`.

### 3. MLX (MacBook M4 Pro)
If you are working locally on your Mac, the script auto-detects the platform:
```bash
python3 train.py
```

---

## 📊 Hardware Expectations (RTX 5080)
| Feature | Status | Benefit |
|---|---|---|
| **BF16 Training** | Enabled | Native Blackwell support; prevents loss overflow. |
| **ZeRO-2** | Enabled | Offloads optimizer states to RAM. |
| **MLA** | Enabled | 4x reduction in KV-cache size vs. standard MHA. |
| **Fused Kernels** | Auto | Combines RMSNorm + Linear for ~15% speedup. |

***

### 📝 Dev Notes (Purdue)
- Keep `num_workers=4` in the `DataLoader` for the 5080 to saturate the PCIe bus.
- If using `ssh`, run in `tmux` or `screen` to prevent signal interrupts during 200k+ step runs.
