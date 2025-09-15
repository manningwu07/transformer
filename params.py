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
    max_len: int = 512
    n_layers: int = 8          # number of transformer blocks 
    lr: float = 1e-5 # Was 3e-4 but too slow, so bumping it up to x2 training speed. 

    # Optimization
    max_epochs: int = 2 # number of passes through the dataset
    patience: int = 16
    improvement_threshold: float = 0.02
    batch_size: int = 8
    epsilon: int = 1e-5
    gradAccumSteps: int = 8 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 5_000
    save_every_steps = 20_000

    # Adam/Optimizer
    warmup_steps: int = 100_000
    decay_steps: int = 300_000
    adam_beta1: float = 0.910
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    weight_decay: float = 0.005
    label_smoothing: float = 0.08
    max_batches: int = 500
    dropout: float = 0.15

    # Debug
    debug: bool = True
    debug_every: int = 1_000     # print every N steps


Config = TrainingConfig()
