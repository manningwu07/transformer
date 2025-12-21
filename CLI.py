import torch
import torch.nn.functional as F
import os
import argparse
from transformer import LLM
from params import Config
from tokenizers import Tokenizer

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_PATH = "data/tokenizer.json" # Path to your trained tokenizer

def load_model(ckpt_path):
    print(f"⏳ Loading model from {ckpt_path}...")
    model = LLM() # Config is pulled automatically from params.py
    
    # Load weights
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully.")
    else:
        print("⚠️ Checkpoint not found! Initializing random model (Gibberish mode).")
    
    model.to(DEVICE).to(dtype=torch.bfloat16) # Use BF16 for inference
    model.eval()
    return model

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded.ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    print(f"\n🤖 AI: ", end="", flush=True)
    
    # Use the generator from the class
    for next_token in model.generate(
        idx, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k,
        use_cache=True
    ):
        decoded_token = tokenizer.decode([next_token.item()])
        print(decoded_token, end="", flush=True)
    
    print("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/latest.pt", help="Path to model checkpoint")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Load Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Tokenizer not found at {TOKENIZER_PATH}. Run tokenizer generation first.")
        return
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Load Model
    model = load_model(args.ckpt)

    print(f"✨ Ready! (Ctrl+C to exit)")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ")
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            
            generate(model, tokenizer, user_input, temperature=args.temp)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()