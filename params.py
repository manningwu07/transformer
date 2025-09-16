# params.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Core transformer parameters (MUST MATCH OR ELSE IT WILL FAIL)
    d_model: int = 768         # model width 
    hidden_size: int = 2048    # MLP hidden dim 
    vocab_size: int = 16_384    # |V| 
    num_heads: int = 2        # attention heads 
    seq_len: int = 16        # context length 
    max_len: int = 512
    n_layers: int = 2          # number of transformer blocks 
    lr: float = 4e-4 # Was 3e-4 but too slow, so bumping it up to x2 training speed. 

    # Optimization
    max_epochs: int = 2 # number of passes through the dataset
    patience: int = 16
    improvement_threshold: float = 0.02
    batch_size: int = 8
    epsilon: int = 1e-5
    gradAccumSteps: int = 8 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 500
    save_every_steps = 20_000

    # Adam/Optimizer
    warmup_steps: int = 50_000
    decay_steps: int = 200_000
    adam_beta1: float = 0.9 #.91
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    weight_decay: float = 0.01 #0.005 
    label_smoothing: float = 0.10 #0.08
    max_batches: int = 250
    dropout: float = 0.10 #0.15

    # Debug
    debug: bool = True
    debug_every: int = 2_000     # print every N steps
    log_random_sample: bool = True
    random_samples: int = 250


Config = TrainingConfig()
