#!/usr/bin/env python3
import requests
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/test/tokenizer.json")
url = "http://127.0.0.1:8000/generate"

while True:
    try:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            break
        # tokenize
        enc = tokenizer.encode(prompt)
        ids = [tokenizer.token_to_id("<bos>")] + enc.ids
        payload = {
            "ids": ids,
            "max_tokens": 50,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.75,
            "repetition_penalty": 1.3,
        }
        r = requests.post(url, json=payload)
        res = r.json()
        print("Raw output:", res)
        print("Bot:", res.get("text"))
    except KeyboardInterrupt:
        break