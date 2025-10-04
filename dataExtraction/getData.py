#!/usr/bin/env python3
# Builds ~TARGET_GB blended_corpus.txt with 50/20/30 category ratios:
# - Broad knowledge/facts: OpenWebText + Wikipedia
# - Conversational: OASST1 + Anthropic HH-RLHF
# - CS/MATH: StackExchange (QA prefs) + GSM8K
#
# Notes:
# - Uses datasets streaming; safe on laptops
# - Adds lightweight structural tags, which we make special tokens later

import os
import argparse
from typing import Dict, Iterator

from datasets import (
    load_dataset,
    interleave_datasets,
    IterableDataset,
    DatasetDict,
    Dataset,
)
from tqdm import tqdm

OUT_PATH = "blended_corpus.txt"
TARGET_GB_DEFAULT = 7.0

# ---------- Formatting ----------
def wrap(tag: str, text: str) -> str:
    text = (text or "").strip()
    return f"<{tag}>\n{text}\n</{tag}>"

def wrap_dialog(user: str, assistant: str) -> str:
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    return f"<dialog>\nUser: {user}\nAssistant: {assistant}\n</dialog>"

def wrap_qa(tag: str, q: str, a: str) -> str:
    return f"<{tag}>\n<Q> { (q or '').strip() }\n<A> { (a or '').strip() }\n</{tag}>"

def best_stack_answer(answers: list) -> str:
    if not answers:
        return ""
    sel = [a for a in answers if a.get("selected")]
    if sel:
        return sel[0].get("text", "")
    return max(answers, key=lambda z: z.get("pm_score", -1)).get("text", "")

# ---------- Sources ----------
def src_c4_broad() -> IterableDataset:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    return ds.map(lambda ex: {"text": wrap("c4", ex.get("text", ""))})

def src_hh_rlhf_stream() -> IterableDataset:
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    return ds.map(lambda ex: {"text": wrap("dialog", ex.get("chosen", ""))})

def src_dolly_iterable() -> IterableDataset:
    base = load_dataset("databricks/databricks-dolly-15k", split="train")
    def gen():
        for ex in base:
            instr = ex.get("instruction", "") or ""
            ctx = ex.get("context", "") or ""
            resp = ex.get("response", "") or ""
            user = instr if not ctx else f"{instr}\n{ctx}"
            yield {"text": wrap_dialog(user, resp)}
    return IterableDataset.from_generator(gen)

def src_oasst_iterable() -> IterableDataset:
    dd: DatasetDict = load_dataset("OpenAssistant/oasst1")
    train: Dataset = dd["train"]
    valid: Dataset = dd.get("validation", None)
    def build_pairs(ds: Dataset):
        prompter = {}
        for ex in ds:
            mid = ex.get("message_id")
            if ex.get("role") == "prompter" and mid:
                prompter[mid] = ex.get("text") or ""
        for ex in ds:
            if ex.get("role") != "assistant":
                continue
            pid = ex.get("parent_id")
            if not pid:
                continue
            user = prompter.get(pid, "")
            if not user:
                continue
            bot = ex.get("text") or ""
            yield {"text": wrap_dialog(user, bot)}
    def gen():
        for it in build_pairs(train):
            yield it
        if valid is not None:
            for it in build_pairs(valid):
                yield it
    return IterableDataset.from_generator(gen)

def src_codeparrot_stream() -> IterableDataset:
    # Clean Python GitHub code (large, easy to stream)
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    return ds.map(lambda ex: {"text": wrap("code", ex.get("content", ""))})

def src_stackexchange_stream() -> IterableDataset:
    ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
    def _map(ex):
        q = (ex.get("question") or "").strip()
        a = best_stack_answer(ex.get("answers", []))
        if not q or not a:
            return {"text": None}
        return {"text": wrap_qa("stack", q, a)}
    return ds.map(_map).filter(lambda ex: ex["text"] is not None)

def src_gsm8k_iterable() -> IterableDataset:
    dd = load_dataset("openai/gsm8k", "main")
    tr = dd["train"]
    def gen():
        for ex in tr:
            yield {"text": wrap_qa("math", ex.get("question", ""), ex.get("answer", ""))}
    return IterableDataset.from_generator(gen)

# ---------- Build blend ----------
def build_blend(target_bytes: int, out_path: str):
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Broad (50%): C4
    broad = src_c4_broad()

    # Conversation (20%): OASST1  Dolly  HH-RLHF (repeat small)
    convo = interleave_datasets(
        [src_oasst_iterable().repeat(1000), src_dolly_iterable().repeat(1000), src_hh_rlhf_stream()],
        probabilities=[0.3, 0.3, 0.4],
        seed=42,
    )

    # CS/Math (30%): CodeParrot  StackExchange  GSM8K (repeat gsm8k)
    cs_math = interleave_datasets(
        [src_codeparrot_stream(), src_stackexchange_stream(), src_gsm8k_iterable().repeat(1000)],
        probabilities=[0.5, 0.35, 0.15],
        seed=42,
    )

    blended = interleave_datasets(
        [broad, convo, cs_math],
        probabilities=[0.5, 0.2, 0.3],
        seed=42,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(blended, desc="Blending → raw text"):
            txt = (ex.get("text") or "").strip()
            if not txt:
                continue
            f.write(txt.replace("\r\n", "\n").replace("\r", "\n") + "\n")
            written += len(txt.encode("utf-8")) + 1
            n += 1
            if written >= target_bytes:
                break
    print(f"✅ Done: {n} lines, ~{written/(1024**3):.2f} GB → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_PATH)
    ap.add_argument("--target_gb", type=float, default=TARGET_GB_DEFAULT)
    args = ap.parse_args()
    build_blend(int(args.target_gb * (1024**3)), args.out)

if __name__ == "__main__":
    main()