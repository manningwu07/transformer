# extract_wikipedia_sentences.py
import os
from datasets import load_dataset
import nltk

# Download sentence tokenizer


def main():
    # Load English Wikipedia dump (latest snapshot)
    print("Loading Wikipedia dataset...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    out_path = "wikipedia_sentences.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, article in enumerate(ds):
            text = article["text"]
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            for s in sentences:
                s = s.strip()
                if len(s) > 0:
                    # Write each sentence as one line
                    f.write(s + "\n")
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1} articles...")

    print(f"✅ Done! Sentences written to {out_path}")


if __name__ == "__main__":
    main()