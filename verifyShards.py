import numpy as np, json
a = np.memmap("data/test/wiki_train_ids-000.bin", dtype=np.int32, mode="r")
ids, counts = np.unique(a[:500000], return_counts=True)
print("Unique IDs:", len(ids))
print("Most common:", sorted(zip(counts, ids), reverse=True)[:30])