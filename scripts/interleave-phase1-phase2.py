from random import random
from dataset import Dataset, PackedBinDataset 

class CurriculumMixDataset(Dataset):
    """
    Mixes two datasets (Phase1/Phase2) with a schedule based on global step.
    
    Schedule:
    - First 0.5B tokens: 50% Phase2 + 50% Phase1
    - After 0.5B tokens: 95% Phase2 + 5% Phase1
    """
    
    def __init__(
        self,
        phase1_ds: PackedBinDataset,
        phase2_ds: PackedBinDataset,
        tokens_per_sample: int,
        warmup_tokens: int = 500_000_000,  # 0.5B
        warmup_phase2_ratio: float = 0.50,
        final_phase2_ratio: float = 0.95,
        seed: int = 1337,
    ):
        self.phase1 = phase1_ds
        self.phase2 = phase2_ds
        self.tokens_per_sample = tokens_per_sample
        self.warmup_samples = warmup_tokens // tokens_per_sample
        self.warmup_phase2_ratio = warmup_phase2_ratio
        self.final_phase2_ratio = final_phase2_ratio
        self.seed = seed
        
        # Total length is Phase2 length (we're training on Phase2 budget)
        self._len = len(phase2_ds)
        
        # Track current position for ratio calculation
        self._global_idx = 0
    
    def set_global_step(self, micro_step: int):
        """Call this from training loop to update curriculum position."""
        self._global_idx = micro_step
    
    def _get_phase2_ratio(self, idx: int) -> float:
        """Linear interpolation from warmup ratio to final ratio."""
        if idx < self.warmup_samples:
            # During warmup: interpolate from warmup_ratio toward final
            progress = idx / max(1, self.warmup_samples)
            return self.warmup_phase2_ratio + progress * (self.final_phase2_ratio - self.warmup_phase2_ratio)
        return self.final_phase2_ratio
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx: int):
        # Use deterministic randomness based on idx
        rng = random.Random(self.seed + idx)
        
        phase2_ratio = self._get_phase2_ratio(self._global_idx + idx)
        
        if rng.random() < phase2_ratio:
            # Sample from Phase2
            p2_idx = idx % len(self.phase2)
            return self.phase2[p2_idx]
        else:
            # Sample from Phase1
            p1_idx = rng.randint(0, len(self.phase1) - 1)
            return self.phase1[p1_idx]