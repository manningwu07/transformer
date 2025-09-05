from datasets import load_dataset
import json

def main():
    # Load Alpaca dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    out_path = "alpaca_conversational.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            instruction = ex["instruction"].strip()
            input_text = ex["input"].strip()
            output_text = ex["output"].strip()

            # Combine instruction + input into "human"
            if input_text:
                human = instruction + "\n" + input_text
            else:
                human = instruction

            # Bot response
            bot = output_text

            record = {
                "human": human,
                "bot": bot
            }
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved reformatted dataset to {out_path}")

if __name__ == "__main__":
    main()