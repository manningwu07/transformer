import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Pure float32 (no autocast)
a = torch.randn(1000, 1000, device=device, dtype=torch.float32, requires_grad=True)
b = torch.randn(1000, 1000, device=device, dtype=torch.float32)
loss = (a @ b).sum()
loss.backward()

print("grad mean/std:", a.grad.mean().item(), a.grad.std().item())