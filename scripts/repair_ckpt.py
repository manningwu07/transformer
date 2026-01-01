#!/usr/bin/env python3
"""repair_rope_ckpt.py - Fix RoPE buffer shape mismatch after max_seq_len change"""

import torch
import sys

def repair_checkpoint(ckpt_path: str, new_max_seq_len: int = 8192):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    state_dict = ckpt["model"]
    modified = []
    
    for key, tensor in state_dict.items():
        # Find all RoPE buffers (cos_cached, sin_cached)
        if "cos_cached" in key or "sin_cached" in key:
            old_shape = tensor.shape
            seq_dim = old_shape[0]  # [max_seq_len, rope_dim//2]
            
            if seq_dim > new_max_seq_len:
                # Truncate to new max_seq_len
                state_dict[key] = tensor[:new_max_seq_len, :]
                modified.append(f"{key}: {old_shape} -> {state_dict[key].shape}")
            elif seq_dim < new_max_seq_len:
                # Need to extend (recompute) - this shouldn't happen in your case
                print(f"⚠️ {key} is SMALLER than target ({seq_dim} < {new_max_seq_len})")
                print("   You'll need to recompute RoPE buffers or use the old max_seq_len")
    
    # Also fix weight tying just in case
    if "tok_embeddings.weight" in state_dict and "output.weight" in state_dict:
        if not torch.equal(state_dict["tok_embeddings.weight"], state_dict["output.weight"]):
            state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
            modified.append("output.weight: re-tied to tok_embeddings.weight")
    
    if modified:
        print("\n✅ Modified tensors:")
        for m in modified:
            print(f"   {m}")
    else:
        print("\n⚠️ No RoPE buffers needed modification. Issue may be elsewhere.")
    
    # Save repaired checkpoint
    out_path = ckpt_path.replace(".pt", "_repaired.pt")
    torch.save(ckpt, out_path)
    print(f"\n💾 Saved to: {out_path}")
    
    # Print summary of key shapes for verification
    print("\n📊 Key tensor shapes in repaired checkpoint:")
    important_keys = [
        "tok_embeddings.weight",
        "output.weight",
        "layers.0.attn.rope.cos_cached",
        "layers.0.attn.rope.sin_cached",
        "layers.0.attn.q_down.weight",
        "layers.0.attn.kv_down.weight",
    ]
    for k in important_keys:
        if k in state_dict:
            print(f"   {k}: {state_dict[k].shape}")

    return out_path


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/best_step_3050.pt"
    new_seq = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    repair_checkpoint(ckpt, new_seq)