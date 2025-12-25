import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from datasets import load_dataset

# 1. Define every special token you will EVER need
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<|endoftext|>",
    "<|user|>", "<|assistant|>", "<|system|>",
    "<thought>", "</thought>", 
    "<call:search>", "<call:tool>", "</call>", "<response>", "</response>",
    "<file_path>", "<file_content>", "</file_content>", "<code>", "</code>",
    "<plan>", "</plan>"
]
def rebuild():
    # 1. Setup BPE
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # 2. Configure for 32k
    trainer = BpeTrainer(
        vocab_size=32768, 
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )

    # 3. Stream from HF (no local corpus needed)
    print("🌊 Streaming SmollM-Corpus for training...")
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    def batch_iterator(batch_size=1000, limit=5_000_000):
        lines = []
        for i, entry in enumerate(ds):
            lines.append(entry["text"])
            if len(lines) >= batch_size:
                yield lines
                lines = []
            if i >= limit: break

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    os.makedirs("data/json", exist_ok=True)
    tokenizer.save("data/json/tokenizer_32k.json")
    print("✅ New 32k Tokenizer saved.")

if __name__ == "__main__":
    rebuild()