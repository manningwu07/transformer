import math
import torch
import torch.nn.functional as F

def evaluate(model, loader, vocab_size, pad_id, device, split="VAL"):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_id,
            )
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    tok_loss = total_loss / total_tokens
    ppl = math.exp(tok_loss)
    print(f"[{split}] tokLoss={tok_loss:.4f}, PPL={ppl:.2f}")
    return tok_loss