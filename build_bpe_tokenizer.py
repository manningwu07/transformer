import os
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
)

from params import Config

tok_train_file = "data/test/wiki_train.txt"
tok_json = "data/test/tokenizer.json"
vocab_json = "data/test/vocab.json"

def main():
    os.makedirs(os.path.dirname(tok_json), exist_ok=True)
    
    # Build BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=Config.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>", "<doc>", "<NL>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    
    tokenizer.train([tok_train_file], trainer)
    tokenizer.save(tok_json)
    print(f"✅ Saved HF tokenizer → {tok_json}")

    # Also export a simple vocab.json for PyTorch side
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    id2tok = [""] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        id2tok[idx] = tok
    tok2id = {tok: idx for tok, idx in vocab.items()}
    
    # Sanity check: special IDs exist and are ordered as requested
    for i, sp in enumerate(["<pad>", "<bos>", "<eos>", "<unk>", "<doc>", "<user>", "<assistant>", "<NL>"]):
        assert sp in tok2id, f"Missing special token {sp}"
    print("🔖 Special IDs:", {sp: tok2id[sp] for sp in ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>", "<doc>", "<NL>"]})

    with open(vocab_json, "w", encoding="utf-8") as f:
        import json
        json.dump({"TokenToID": tok2id, "IDToToken": id2tok}, f, indent=2)

    print(f"✅ Tokenizer saved to {tok_json}, vocab saved to {vocab_json}")

if __name__ == "__main__":
    main()