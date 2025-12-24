import os
import json
import sys
import time
import threading
import concurrent.futures
import requests
from dotenv import load_dotenv

instance_id = sys.argv[1] if len(sys.argv) > 1 else "1"
OUTPUT_FILE = f"data/raw/synthetic_reasoning_{instance_id}.jsonl"

load_dotenv()
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
# Using Gemini Flash for the "Chaos" generation (Fast/Cheap)
CHAOS_MODEL = "google/gemini-2.0-flash-001"
# Using DeepSeek V3 for the "Compression" spec
COMPRESSOR_MODEL = "deepseek/deepseek-v3.2"
WORKERS = 10  # Fan out for DeepSeek
TARGET_COUNT = 250 # Total pairs to generate (10k)

# System prompt for the Compressor
COMPRESS_SYSTEM_PROMPT = (
    "You are a high-fidelity technical state-machine. "
    "Your task is to take a messy human rant and compress it into a 'Structured Intent Spec' without losing ANY constraints mentioned. "
    "Output ONLY the compressed string. No intro, no markdown, no JSON, no fluff. "
    "Rules: Ignore filler, resolve ambiguity to hard specs, use pipes '|' for options, "
    "commas for lists, and key=value pairs. Abbreviate: ctx, bs, lr, qps, etc."
)

# The prompt you provided, optimized for 50 examples per batch
CHAOS_PROMPT = """
Generate 50 diverse, stream-of-consciousness project ideas formatted as messy human input. 
Each should be 300-500 tokens. Style: technical jargon mixed with casual language, tangents, asides, questions, uncertainty. 
Format: Plain text. Separate each example with ===== on its own line.

CRITICAL RULES:
1. NO logical flow. Start in the middle of a thought.
2. SELF-CORRECTION: Mid-sentence, the person should realize an idea is stupid and change it. 
   (e.g., "Wait, no, React is too heavy for this... let's use HTMX instead. Actually, maybe just raw C?")
3. AMBIGUITY: Leave some things unresolved. "Should I use X or Y? Idk, I'll figure it out later."

Rules:
- NO numbered lists. NO bullet points.
- DO NOT say "Here are some ideas" or "Sure, I can help."
- Use 80% lowercase. Mix in some typos (e.g., "dosnt", "thign", "prolly").
- The thoughts should be a "wall of text" with occasional ellipses (...) or line breaks.
- Separate each 500-token monologue with EXACTLY: =====
- DOMAINS: 40% Engineering, 25% Business, 20% Cooking, 15% Creative/Lifestyle.

Examples: 
"Okay so I want to build this thing that basically takes user input and converts it to structured output, right? Like a parser but smarter. Maybe use transformers? But idk if I need a massive model or if something 100M params would work. I have like $500 budget and 3 weeks. Data is messy, probably need to scrape it from... somewhere? Reddit? HuggingFace? Also need inference to be fast, like under 100ms per request. Should I use quantization or just go full bfloat16? And do I deploy on cloud or local GPU?"
"Alright so I'm trying to make this restaurant-quality pasta from scratch but like... I don't have a pasta maker? Can I use a rolling pin? And idk the exact ratios for the dough, like should it be wetter or drier? Also trying to make like 5 different sauces to go w it but I'm worried they'll all taste similar or something. Maybe I'm overthinking this. Budget is like $20 and I have 2 hours before guests arrive. Should I just buy pasta ngl."
"Hey so I'm thinking of starting this side hustle where I design custom t-shirts but like... I have zero design skills lol. Maybe I can use some AI tool to generate designs? But then how do I handle printing and shipping? Also not sure how to price them, like do I go for premium or budget? And marketing is a whole other can of worms, should I just do Instagram ads or try TikTok? Budget is tight, maybe $100 to start. Any ideas on how to make this work?"
"Yo so I wanna build this app that helps people track their fitness goals but like... there are a million apps out there already, right? How do I make mine stand out? Maybe focus on mental health too? But idk how to integrate that without making it too complicated. Also need to think about UX/UI, I'm not a designer lol. Budget is like $2000 and timeline is 2 months. Should I learn to code it myself or hire someone? And what platform, iOS or Android or both? So many questions..."
"Okay so I have this idea for a novel but it's like... sci-fi mixed with romance? And I don't really know how to structure it. Should I write the romance first then layer the sci-fi worldbuilding on top? Or is that backwards? I'm terrible at planning stories. Also I don't have much time, maybe 2-3 months to write a draft? And I'm not even sure if the premise is good tbh. Like, would ppl actually care about a story w aliens and... idk, maybe love triangles?"
"So I want to get fit right? But like... I hate the gym. Maybe I should just do home workouts? Or running? But it's cold outside rn and I live in an apt so can't make noise. I have like 45 mins a day max. Should I follow a program or just wing it? Also my diet is shit, should I fix that first or start working out? Idk man, motivation is hard lol."

"""

def call_openrouter(model: str, system: str, user: str, temp: float = 1.0):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temp,
        "include_reasoning": False
    }
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error calling {model}: {e}")
        return None

file_lock = threading.Lock()
def process_single_example(messy_text: str):
    """DeepSeek Compression Step"""
    spec = call_openrouter(COMPRESSOR_MODEL, COMPRESS_SYSTEM_PROMPT, messy_text, temp=0.1)
    if spec:
        entry = {
            "instruction": "Compress this project idea into a dense planning spec.",
            "input": messy_text,
            "output": spec
        }
        # Thread-safe append to JSONL
        with file_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return True
    return False

def main():
    if not API_KEY:
        print("Set OPEN_ROUTER_API_KEY environment variable.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    total_generated = 0
    
    print(f"🔥 Burning through $15... Target: {TARGET_COUNT} pairs.")

    while total_generated < TARGET_COUNT:
        print(f"📦 Fetching batch of chaos from {CHAOS_MODEL}...")
        batch_text = call_openrouter(CHAOS_MODEL, "You are a creative writer.", CHAOS_PROMPT, temp=1.2)
        
        if not batch_text:
            time.sleep(5)
            continue
            
        # Split by the delimiter
        examples = [ex.strip() for ex in batch_text.split("=====") if len(ex.strip()) > 50]
        print(f"⚡ Found {len(examples)} examples. Compressing with {WORKERS} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            results = list(executor.map(process_single_example, examples))
        
        count = sum(1 for r in results if r)
        total_generated += count
        print(f"✅ Progress: {total_generated}/{TARGET_COUNT} | Batches of Chaos left: {(TARGET_COUNT-total_generated)//50}")

if __name__ == "__main__":
    main()