import math
import random
import torch


def evaluate(
    model,
    loader,
    vocab_size,
    pad_id,
    device,
    split="VAL",
    max_batches=None,
    shard_frac=1.0,
):
    """
    Evaluate model on a subset of shards.
    Args:
      max_batches: if set (int), evaluate only this many batches total.
      shard_frac: fraction of shards to use (1.0 = all).
    """
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    # Build sub-loader if we can access shards; otherwise, use the provided loader.
    use_subset = hasattr(loader, "dataset") and hasattr(loader.dataset, "shards")
    if use_subset:
        all_shards = list(loader.dataset.shards)
        n_total = len(all_shards)
        n_selected = max(1, int(round(n_total * float(shard_frac))))
        if n_selected >= n_total:
            selected = all_shards
        else:
            selected = random.sample(all_shards, n_selected)

        sub_ds = loader.dataset.__class__(
            loader.dataset.prefix, loader.dataset.seq_len, repeat=False, shuffle=False
        )
        sub_ds.shards = selected
        num_workers = getattr(loader, "num_workers", 0) or 0
        sub_loader = torch.utils.data.DataLoader(
            sub_ds, batch_size=loader.batch_size, num_workers=num_workers
        )
        frac_effective = len(selected) / max(1, n_total)
        print(
            f"[{split}] Using {len(selected)}/{n_total} shards (~{frac_effective*100:.1f}%). "
            f"Eval up to {max_batches or 'ALL'} batches"
        )
    else:
        sub_loader = loader
        print(f"[{split}] Evaluating loader (no shard control).")

    batches_run = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(sub_loader):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss_sum = criterion(logits.view(-1, vocab_size), y.view(-1)).item()
            tokens = (y != pad_id).sum().item()

            total_loss_sum += loss_sum
            total_tokens += tokens

            batches_run += 1
            if max_batches is not None and batches_run >= int(max_batches):
                break
            if i % 100 == 0:
                print(
                    f"[{split}] {batches_run} batches, {total_tokens} non-pad tokens",
                    end="\r",
                )

    avg_loss = total_loss_sum / max(1, total_tokens)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    print(f"\n[{split}] tokLoss={avg_loss:.4f}, PPL={ppl:.2f}")
    return avg_loss