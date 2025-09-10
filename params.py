# params.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Core transformer parameters (MUST MATCH OR ELSE IT WILL FAIL)
    d_model: int = 768         # model width 
    hidden_size: int = 2048    # MLP hidden dim 
    vocab_size: int = 16_384    # |V| 
    num_heads: int = 12        # attention heads 
    seq_len: int = 512        # context length 
    n_layers: int = 8          # number of transformer blocks 
    lr: float = 3e-4

    # Optimization
    max_epochs: int = 2 # number of passes through the dataset
    patience: int = 32
    improvement_threshold: float = 0.005
    batch_size: int = 4
    epsilon: int = 1e-5
    gradAccumSteps: int = 4 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 25_000
    save_every_steps = 100_000

    # Adam/Optimizer
    warmup_steps: int = 10_000
    decay_steps: int = 1_000_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # Debug
    debug: bool = True
    debug_every: int = 1_000     # print every N steps


Config = TrainingConfig()