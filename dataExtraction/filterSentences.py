import re

def is_high_quality(s):
    # Length filter
    n = len(s.split())
    if n < 5 or n > 25:
        return False
    # Must end with a period
    if not s.endswith("."):
        return False
    # Drop citations, tables, weird formatting
    if re.search(r"\[\d+\]", s):
        return False
    if re.match(r"^\d", s):
        return False
    return True

# Load your extracted sentences
with open("wikipedia_sentences.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Filter
filtered = [s for s in sentences if is_high_quality(s)]

# Save top N (say 5M)
with open("wikipedia_high_quality.txt", "w", encoding="utf-8") as f:
    for s in filtered[:5_000_000]:
        f.write(s + "\n")

print(f"✅ Saved {len(filtered[:5_000_000])} high-quality sentences")