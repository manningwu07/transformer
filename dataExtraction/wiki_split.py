import random

with open("wikipedia_high_quality.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

random.shuffle(sentences)

split = int(0.95 * len(sentences))
train = sentences[:split]
eval_set = sentences[split:]

with open("wiki_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train))

with open("wiki_eval.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(eval_set))

print(f"Train: {len(train)} sentences, Eval: {len(eval_set)} sentences")