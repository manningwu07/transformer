#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PackedBinDataset
from params import Config, TrainCfg
from transformer import LLM
from utils import validate


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    # Avoid float8 conversion in eval unless you explicitly want it.
    Config.use_float8 = False

    model = LLM(Config).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Your CheckpointManager saves payload["model"].
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Normalize compiled key prefix if present.
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--val_dir", required=True, type=str)
    ap.add_argument("--pattern", default="*val-*.bin", type=str)
    ap.add_argument("--seq_len", default=TrainCfg.seq_len, type=int)
    ap.add_argument("--batch_size", default=TrainCfg.batch_size, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--max_val_steps", default=200, type=int)
    ap.add_argument(
        "--dtype",
        default="u16",
        choices=["u16", "u32"],
        type=str,
        help="Token dtype of shards",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard_dtype = np.uint16 if args.dtype == "u16" else np.uint32

    model = load_model_from_ckpt(args.ckpt, device)

    ds = PackedBinDataset(
        args.val_dir,
        split="val",
        seq_len=args.seq_len,
        dtype=shard_dtype,
        pattern=args.pattern,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    loss, ppl = validate(model, device, dl, max_val_steps=args.max_val_steps)
    print(f"ckpt={args.ckpt}")
    print(f"val_dir={args.val_dir} pattern={args.pattern}")
    print(f"loss={loss:.4f} ppl={ppl:.2f}")


if __name__ == "__main__":
    main()