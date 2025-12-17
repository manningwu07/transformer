#!/usr/bin/env python3
import argparse
import json
import os
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor

# --- CRITICAL: Special Tokens for 1B Planner ---
# We add structural tokens so the model learns to output them as single atomic units.
# This stabilizes the "thinking" process.
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    # Structure
    "<|user|>", "<|assistant|>", "<|system|>",
    # Thinking & Planning (DeepSeek/O1 style)
    "<plan>", "</plan>", 
    "", 
    "<code_block>", "</code_block>",
    # Data Sources (from your blending script)
    "<math>", "</math>", "<python>", "</python>"
]

def train_tokenizer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/raw/corpus.txt")
    parser.add_argument("--out", type=str, default="data/vocab")
    parser.add_argument("--vocab_size", type=int, default=65536) # 2^16 is GPU friendly
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Initialize BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # 2. Normalization (NFKC is standard, but ByteLevel handles raw bytes better for code)
    # We stick to ByteLevel pre-tokenization which is standard for Llama/GPT
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # 3. Trainer
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2, # Filter out ultra-rare noise
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )

    # 4. Train
    print(f"🚂 Training Tokenizer on {args.corpus}...")
    # We assume corpus.txt is a massive file. Tokenizers library handles it well,
    # but for 100GB+ files, you might want to stream random chunks. 
    # For <50GB, this is fine on most machines.
    tokenizer.train([args.corpus], trainer)

    # 5. Post-Processing
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 6. Save
    # Save the full tokenizer object (fastest to load)
    tok_path = os.path.join(args.out, "tokenizer.json")
    tokenizer.save(tok_path)
    
    # Save legacy vocab.json for manual inspection/debugging
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(args.out, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"✅ Tokenizer saved to {args.out}")
    print(f"📊 Vocab Size: {tokenizer.get_vocab_size()}")
    print("Example Special IDs:", [tokenizer.token_to_id(t) for t in ["<plan>", "<math>"]])

if __name__ == "__main__":
    train_tokenizer()