import argparse
import os
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

def get_phase_1_stream():
    """Phase 1: 50B tokens (SmolLM Corpus)"""
    print("🌊 Loading SmolLM Corpus (Cosmopedia + FineWeb-Edu + Python)...")
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    # Filter or map if necessary
    return ds.map(lambda x: {"text": x["text"] + "\n<|endoftext|>\n"})

def get_phase_2_stream():
    """Phase 2: Math + Reasoning + Synthetic Planning"""
    print("📐 Loading FineMath & Synthetic Data...")
    
    # 1. FineMath (47B allocation)
    finemath = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
    finemath = finemath.map(lambda x: {"text": f"<math>\n{x['text']}\n</math>\n"})
    
    # 2. Synthetic Planning (1B allocation -> Duplicated)
    # Assumes you ran planner_script.py and have a JSONL/Text file
    if os.path.exists("data/synthetic_planning.json"):
        planning = load_dataset("json", data_files="data/synthetic_planning.json", split="train", streaming=True)
        # Duplicate planning data heavily to meet token count
        planning = planning.repeat(1000) 
        planning = planning.map(lambda x: {"text": f"<plan>\nInstruction: {x['instruction']}\nInput: {x['input']}\nPlan: {x['output']}\n</plan>\n"})
    else:
        print("⚠️ Warning: planning data not found, falling back to just math")
        planning = finemath # Fallback
        
    return interleave_datasets([finemath, planning], probabilities=[0.95, 0.05])

def get_phase_3_stream():
    """
    Phase 3: SFT (Supervised Fine-Tuning) + DPO (Direct Preference Optimization) preparation.
    We format DPO data as 'chosen' samples for SFT (to learn the good response first).
    """
    print("🧠 Loading Phase 3: SFT + Long Context Mix...")

    # 1. High-Quality Instructions (SFT) - e.g., OpenHermes or Magpie
    sft_ds = load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", split="train", streaming=True)
    
    # 2. DPO Data (Intel/orca_dpo_pairs) - We use the 'chosen' response for SFT
    dpo_ds = load_dataset("Intel/orca_dpo_pairs", split="train", streaming=True)

    # 3. Long Context (PG-19 or similar for maintaining 8k ability)
    long_ds = load_dataset("pg19", split="train", streaming=True)

    def format_sft(ex):
        # Standard user/assistant format
        convo = ex.get('conversations', [])
        text = "\n".join([f"<|{msg['role']}|>\n{msg['value']}<|end|>" for msg in convo])
        return {"text": f"<|begin_of_text|>{text}"}

    def format_dpo_as_sft(ex):
        # Train on the 'chosen' response to reinforce good behavior
        prompt = ex.get('system', '') + "\n" + ex.get('question', '')
        chosen = ex.get('chosen', '')
        return {"text": f"<|begin_of_text|><|user|>\n{prompt}<|end|>\n<|assistant|>\n{chosen}<|end|>"}

    sft_stream = sft_ds.map(format_sft)
    dpo_stream = dpo_ds.map(format_dpo_as_sft)
    long_stream = long_ds.map(lambda x: {"text": f"<|long_ctx|>{x['text'][:30000]}<|end|>"}) # Truncate to reasonable length

    # Mix: 70% SFT, 20% DPO-Chosen, 10% Long Context
    return interleave_datasets([sft_stream, dpo_stream, long_stream], probabilities=[0.7, 0.2, 0.1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="phase1")
    parser.add_argument("--out", type=str, default="data/raw/corpus.txt")
    parser.add_argument("--limit_gb", type=float, default=100.0)
    args = parser.parse_args()
    
    if args.phase == "phase1":
        ds = get_phase_1_stream()
    elif args.phase == "phase2":
        ds = get_phase_2_stream()
    else:
        # Phase 3 logic (Long context mix)
        ds = get_phase_1_stream() # Placeholder
        
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    written_bytes = 0
    limit = args.limit_gb * 1024**3
    
    with open(args.out, "w") as f:
        for ex in tqdm(ds, desc=f"Extracting {args.phase}"):
            txt = ex.get("text", "").strip()
            if not txt: continue
            
            f.write(txt + "\n")
            written_bytes += len(txt.encode('utf-8'))
            
            if written_bytes > limit:
                break
    print(f"✅ Done. Saved to {args.out}")

if __name__ == "__main__":
    main()