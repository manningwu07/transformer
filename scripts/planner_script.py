import os
import requests
import json
import time

# --- CONFIG ---
API_KEY = "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" # DO NOT LEAK ENV VARS
INPUT_FILE = "user_plans_prompts.txt"
OUTPUT_FILE = "bot_plan_response.json"
MODEL = "deepseek/deepseek-v3.2" # DeepSeek V3.2

# System prompt optimized for RAW string output only
SYSTEM_PROMPT = (
    "You are a context compression engine. Extract executable specs from messy human input. "
    "Output ONLY the compressed string. No intro, no markdown, no JSON, no fluff. "
    "Rules: Ignore filler, resolve ambiguity to hard specs, use pipes '|' for options, "
    "commas for lists, and key=value pairs. Abbreviate: ctx, bs, lr, qps, etc."
)

def get_compressed_spec(messy_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": messy_text}
        ],
        "temperature": 0.1 # Low temp for consistency
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                 headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error calling DeepSeek: {e}")
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        # Split by your delimiter
        raw_prompts = [p.strip() for p in f.read().split("=====") if p.strip()]

    dataset = []
    print(f"Starting synthesis for {len(raw_prompts)} prompts...")

    for i, prompt in enumerate(raw_prompts):
        print(f"[{i+1}/{len(raw_prompts)}] Processing...")
        
        compressed_spec = get_compressed_spec(prompt)
        
        if compressed_spec:
            # Code handles the JSON construction as requested
            data_entry = {
                "instruction": "Compress this project idea into a dense planning spec.",
                "input": prompt,
                "output": compressed_spec
            }
            dataset.append(data_entry)
            
            # Save incrementally in case of crash/timeout
            with open(OUTPUT_FILE, "w") as f:
                json.dump(dataset, f, indent=2)
        
        time.sleep(0.3) # Avoid hitting rate limits too hard

    print(f"Done. Synthetic data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()