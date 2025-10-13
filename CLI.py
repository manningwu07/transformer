#!/usr/bin/env python3
from tokenizers import Tokenizer
import argparse, requests

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_tokens", type=int, default=80)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rep_pen", type=float, default=1.2)
    ap.add_argument("--tok", type=str, default="data/json/tokenizer.json")
    return ap.parse_args()

args = parse_args()
tokenizer = Tokenizer.from_file(args.tok)
url = f"http://{args.host}:{args.port}/generate"
print(f"Connecting to {url}")

while True:
    try:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue
        # tokenize
        enc = tokenizer.encode(prompt)
        ids = [tokenizer.token_to_id("<bos>")] + enc.ids
        payload = {
            "ids": ids,
            "max_tokens": args.max_tokens,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.rep_pen,
        }
        r = requests.post(url, json=payload, timeout=30)
        res = r.json()
        if "error" in res:
            print("[SERVER ERROR]", res["error"])
        else:
            print(f"Bot: {res.get('text', '').strip()}")
        # Uncomment for debug:
        print(res)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print("[CLI ERROR]", e)
        continue