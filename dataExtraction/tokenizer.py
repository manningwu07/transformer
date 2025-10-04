#!/usr/bin/env python3
import argparse
import json
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Core + structure/control tokens used in the blend
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<NL>",
    # semantics
    "<c4>", "</c4>", "<oscar>", "</oscar>",
    # conversation
    "<dialog>", "</dialog>",
    # code / qa / math
    "<code>", "</code>", "<stack>", "</stack>",
    "<math>", "</math>", "<mathqa>", "</mathqa>", "<comp_math>", "</comp_math>",
    # generic QA tags
    "<Q>", "<A>",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default="data/raw/blended_corpus.txt")
    p.add_argument("--out", type=str, default="data/json")
    p.add_argument("--vocab_size", type=int, default=65536)
    p.add_argument("--min_freq", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"Training tokenizer on {args.corpus} ...")
    tok.train([args.corpus], trainer)

    tok_path = os.path.join(args.out, "tokenizer.json")
    tok.save(tok_path)
    print(f"Saved tokenizer → {tok_path}")

    # vocab.json your training code expects
    vocab = tok.get_vocab()  # token -> id
    tok2id = {k: int(v) for k, v in vocab.items()}
    id2tok = [None] * len(tok2id)
    for t, i in tok2id.items():
        id2tok[i] = t

    vocab_path = os.path.join(args.out, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"TokenToID": tok2id, "IDToToken": id2tok}, f)
    print(f"Saved vocab map → {vocab_path}")

    print("Special IDs:", {t: tok.token_to_id(t) for t in SPECIAL_TOKENS})
    
    # === Collect "bad" tokens (special & delimiters) for generation masking ===
    print("Scanning special tokens to build bad_id list...")

    bad_ids = []
    # Iterate only over the first chunk of tokens (specials are front‑loaded)
    for i, (tok_str, tok_id) in enumerate(tok.get_vocab().items()):
        if i >= 1024:  # skip normal tokens to save time
            break
        if tok_str.startswith("<") and tok_str.endswith(">") and len(tok_str) > 2:
            bad_ids.append(tok_id)

    # Also always mask the base special tokens if they exist
    for special in ("<pad>", "<unk>", "<bos>"):
        try:
            tok_id = tok.token_to_id(special)
            if tok_id is not None:
                bad_ids.append(tok_id)
        except KeyError:
            pass

    # Deduplicate and sort
    bad_ids = sorted(set(bad_ids))

    # Save alongside tokenizer and vocab files
    bad_path = os.path.join(args.out, "data/json/bad_ids.json")
    with open(bad_path, "w") as f:
        json.dump({"BadTokenIDs": bad_ids}, f, indent=2)

    print(f"✅ Saved bad token IDs to {bad_path} ({len(bad_ids)} entries)")
if __name__ == "__main__":
    main()