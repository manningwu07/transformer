import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 1. Define every special token you will EVER need
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<|endoftext|>",
    "<|user|>", "<|assistant|>", "<|system|>",
    "<thought>", "</thought>", 
    "<call:search>", "<call:tool>", "</call>", "<response>", "</response>",
    "<file_path>", "<file_content>", "</file_content>", "<code>", "</code>",
    "<plan>", "</plan>"
]

def train():
    # Setup BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=65535, # GPU friendly 2^16
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )

    # Iterator to read the file line by line without crashing RAM
    def batch_iterator(path, total_lines_to_sample=5_000_000):
        # This skips lines to ensure we see the beginning, middle, and end of the 53GB
        # Estimate total lines in 53GB is roughly 500M. So we sample every 100th line.
        skip_step = 100 
        
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % skip_step == 0:
                    yield line
                    count += 1
                if count >= total_lines_to_sample:
                    break

    print("🛠️ Training tokenizer on sampled Phase 1 corpus...")
    tokenizer.train_from_iterator(
        batch_iterator("data/raw/phase1_corpus.txt"), 
        trainer=trainer
    )

    os.makedirs("data/json", exist_ok=True)
    tokenizer.save("data/json/tokenizer.json")
    print("✅ Tokenizer saved to data/json/tokenizer.json")

if __name__ == "__main__":
    train()