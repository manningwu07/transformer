# # LORA

# from dataclasses import dataclass

# @dataclass
# class TrainingConfig:
#     # Core model architecture must match pretraining checkpoint
#     d_model: int = 768
#     hidden_size: int = 2048
#     vocab_size: int = 65_536
#     num_heads: int = 12
#     seq_len: int = 64
#     max_len: int = 512
#     n_layers: int = 12

#     # === Finetuning hyperparameters (LoRA) ===
#     lr: float = 1e-3           # higher LR, since only LoRA adapters train
#     batch_size: int = 4        # keep modest for stability on MPS
#     gradAccumSteps: int = 8    # effective batch_size = 32 (4*8)
#     max_steps: int = 3000      # LoRA converges fast, 1-3k is often enough
#     warmup_steps: int = 100    # much shorter warmup than pretrain
#     decay_steps: int = 3000    # cosine decay for full fine-run
#     epsilon: float = 1e-8
#     grad_clip: float = 1.0

#     # === Regularization (keep minimal in finetune) ===
#     dropout: float = 0.0       # better to switch off for instruction tuning
#     label_smoothing: float = 0.0
#     patience: int = 5          # earlier stop allowed
#     improvement_threshold: float = 0.002

#     # Debug/logging
#     debug: bool = True
#     debug_every: int = 50
#     log_random_sample: bool = False
#     random_samples: int = 100

# Config = TrainingConfig()

# params.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Core transformer parameters (MUST MATCH OR ELSE IT WILL FAIL)
    d_model: int = 768         # model width 
    hidden_size: int = 2048    # MLP hidden dim 
    vocab_size: int = 65_536   # |V| 
    num_heads: int = 12 # attention heads 
    seq_len: int = 64        # context length 
    max_len: int = 512
    n_layers: int = 12          # number of transformer blocks 
    lr: float = 1e-3 # 3e-3 for finetuning (LoRA)

    # Optimization
    max_epochs: int = 5 # number of passes through the dataset
    patience: int = 10
    improvement_threshold: float = 0.02
    batch_size: int = 16
    epsilon: int = 1e-5
    gradAccumSteps: int = 16 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 2_000
    max_batches: int = 250
    save_every_steps = 10_000
    label_smoothing: float = 0.0
    dropout: float = 0.0

    # AdaFactor
    warmup_steps: int = 1_000
    decay_steps: int = 5_000
    grad_clip: float = 1.0

    # Debug
    debug: bool = True
    debug_every: int = 500     # print every N steps
    log_random_sample: bool = False
    random_samples: int = 250


Config = TrainingConfig()
