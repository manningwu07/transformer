#!/usr/bin/env python3
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import json

train_file = "data/raw/wiki_train.txt"   # your large corpus
tok_json = "data/test/tokenizer.json"
vocab_json = "data/test/vocab.json"

def main():
    # Build BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    # Train
    tokenizer.train([train_file], trainer)

    # Save HF JSON
    tokenizer.save(tok_json)
    print(f"✅ Saved HF tokenizer → {tok_json}")

    # Also export a simple vocab.json for PyTorch side
    vocab = tokenizer.get_vocab()
    id2tok = [None] * len(vocab)
    for tok, idx in vocab.items():
        id2tok[idx] = tok
    tok2id = {tok: idx for tok, idx in vocab.items()}
    with open(vocab_json, "w") as f:
        json.dump({"TokenToID": tok2id, "IDToToken": id2tok}, f, indent=2)
    print(f"✅ Saved vocab.json → {vocab_json}")

if __name__ == "__main__":
    main()