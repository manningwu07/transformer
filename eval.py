import math
import random
import torch
import torch.nn.functional as F

def evaluate(model, loader, vocab_size, pad_id, device, split="VAL", subset=1.0):
    """
    Evaluate model on a subset fraction of shards.
    subset=1.0 → full validation set
    subset=0.1 → randomly sample ~10% of shards
    """
    assert 0 < subset <= 1.0
    model.eval()

    # Get shard list from dataset (works for IndexedBinaryDataset)
    all_shards = loader.dataset.shards
    n_selected = max(1, int(len(all_shards) * subset))
    selected = random.sample(all_shards, n_selected)

    print(f"[{split}] Using {n_selected}/{len(all_shards)} shards (~{subset*100:.1f}%)")

    total_loss, total_tokens = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    with torch.no_grad():
        for shard in selected:
            for x, y in loader.dataset._iter_shard(shard):
                x, y = x.to(device), y.to(device)
                logits, _ = model(x.unsqueeze(0))  # add batch dim
                loss = criterion(
                    logits.view(-1, vocab_size), 
                    y.view(-1)
                )
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()

    tok_loss = total_loss / total_tokens
    ppl = math.exp(tok_loss)
    print(f"[{split}] tokLoss={tok_loss:.4f}, PPL={ppl:.2f}")
    return tok_loss