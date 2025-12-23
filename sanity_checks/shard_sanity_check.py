from tokenizers import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_file("data/json/tokenizer.json")
# Load the first 100 tokens from your first shard
data = np.fromfile("data/shards/phase1/phase1-train-000.bin", dtype=np.uint16, count=500)
print(tokenizer.decode(data))