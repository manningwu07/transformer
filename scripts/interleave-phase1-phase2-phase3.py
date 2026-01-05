# ~/dataset.py

import random
from torch.utils.data import Dataset

class Phase3MixDataset(Dataset):
    def __init__(
        self,
        p1_ds: Dataset, # Phase 1 Shards
        p2_ds: Dataset, # Phase 2 Shards
        p3_ds: Dataset, # Phase 3 Shards
        tokens_per_sample: int,
        total_tokens: int = 5_000_000_000,
        seed: int = 1337
    ):
        self.p1 = p1_ds
        self.p2 = p2_ds
        self.p3 = p3_ds
        self.total_samples = total_tokens // tokens_per_sample
        self.tokens_per_sample = tokens_per_sample
        self.seed = seed
        self._current_step = 0

    def set_step(self, step: int):
        self._current_step = step

    def _get_ratios(self):
        # Progress calculation (0.0 to 1.0)
        progress = self._current_step / max(1, self.total_samples)
        
        if progress < 0.2: # First 1B tokens (Warmup)
            return 0.60, 0.30, 0.10
        elif progress < 0.8: # Middle 3B tokens (Core)
            return 0.85, 0.10, 0.05
        else: # Final 1B tokens (Cool down)
            return 0.95, 0.04, 0.01

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Deterministic but varied mixing
        rng = random.Random(self.seed + idx + self._current_step)
        r3, r2, r1 = self._get_ratios()
        
        choice = rng.random()
        
        if choice < r3:
            return self.p3[idx % len(self.p3)]
        elif choice < (r3 + r2):
            return self.p2[rng.randint(0, len(self.p2) - 1)]
        else:
            return self.p1[rng.randint(0, len(self.p1) - 1)]