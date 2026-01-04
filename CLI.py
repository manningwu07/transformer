# CLI.py
import os
import torch
import sys
from tokenizers import Tokenizer
from transformer import LLM
from params import Config

# --- CONFIGURATION ---
# Change this to point to your latest/best checkpoint
CHECKPOINT_PATH = "models/ckpt_step_4000.pt" 
TOKENIZER_PATH = "data/json/tokenizer_32k.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(ckpt_path):
    print(f"📦 Loading model from {ckpt_path}...")
    
    # Initialize model with training config
    # We disable float8 for inference to avoid torchao overhead in CLI
    Config.use_float8 = False 
    model = LLM(Config).to(DEVICE)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    # CheckpointManager saves everything in a "model" key
    state_dict = ckpt["model"]
    
    # Handle the 'compiled' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Error: Tokenizer not found at {TOKENIZER_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = load_model(CHECKPOINT_PATH)
    
    print("\n" + "="*50)
    print("🚀 1B MLA MODEL - PHASE 1 INFERENCE")
    print("Format: Base Completion (Acts like Autocomplete)")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        prompt = input("\nPrompt > ")
        if prompt.lower() in ["exit", "quit"]:
            break
            
        if not prompt.strip():
            continue

        # Inference Params
        max_new_tokens = 128
        temp = 0.8
        top_k = 40

        # Encode
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
        
        print("\nCompletion: ", end="", flush=True)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # The generate method is a generator (yields tokens)
            for token_tensor in model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens, 
                temperature=temp, 
                top_k=top_k,
                use_cache=True
            ):
                token_id = token_tensor.item()
                # Stop if model generates <|endoftext|> (ID typically 0 or in config)
                if token_id == tokenizer.token_to_id("<|endoftext|>"):
                    break
                    
                word = tokenizer.decode([token_id])
                print(word, end="", flush=True)
        
        print("\n" + "-"*30)

if __name__ == "__main__":
    main()