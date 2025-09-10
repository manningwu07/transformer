import json
import random

def split_alpaca(jsonl_path="alpaca_conversational.jsonl",
                 train_out="alpaca_train.jsonl",
                 eval_out="alpaca_eval.jsonl",
                 eval_frac=0.05,
                 seed=42):
    # Load all records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(data)

    # Split
    split_idx = int(len(data) * (1 - eval_frac))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # Save train
    with open(train_out, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    # Save eval
    with open(eval_out, "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Split complete: {len(train_data)} train, {len(eval_data)} eval")

if __name__ == "__main__":
    split_alpaca()