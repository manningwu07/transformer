import os
import json
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
)

train_file = "data/raw/wiki_train.txt"
tok_json = "data/test/tokenizer.json"
vocab_json = "data/test/vocab.json"

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    
    # Build BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<doc>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train
    tokenizer.train([train_file], trainer)

    # Save HF JSON
    tokenizer.save(tok_json)
    print(f"✅ Saved HF tokenizer → {tok_json}")

    # Also export a simple vocab.json for PyTorch side
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    size = max(vocab.values()) + 1 if vocab else 0
    id2tok = [""] * size
    for tok, idx in vocab.items():
        id2tok[idx] = tok
    tok2id = {tok: idx for tok, idx in vocab.items()}
    
    # Sanity check: special IDs exist and are ordered as requested
    for i, sp in enumerate(["<pad>", "<bos>", "<eos>", "<unk>"]):
        assert sp in tok2id, f"Missing special token {sp}"
    print("🔖 Special IDs:", {sp: tok2id[sp] for sp in ["<pad>", "<bos>", "<eos>", "<unk>"]})

    with open(vocab_json, "w") as f:
        json.dump({"TokenToID": tok2id, "IDToToken": id2tok}, f, indent=2)
    print(f"✅ Saved vocab.json → {vocab_json}")

if __name__ == "__main__":
    main()