#!/usr/bin/env python3
# build_wiki_corpus.py
import argparse
import hashlib
import os
import re
from typing import Iterable, List

from datasets import load_dataset
import nltk


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt")


def clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sent_ok(s: str, min_chars: int = 20, min_alpha_ratio: float = 0.6) -> bool:
    s = s.strip()
    if len(s) < min_chars:
        return False
    alpha = sum(ch.isalpha() for ch in s)
    ratio = alpha / max(1, len(s))
    if ratio < min_alpha_ratio:
        return False
    return True


def chunk_by_chars(text: str, max_chars: int = 3000) -> list[str]:
    """
    Greedy split of article into ~max_chars chunks.
    Always respects max_chars upper bound.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space != -1 and space > start:
                end = space   # prefer to cut at last space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def split_bucket(title: str, eval_frac: float, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{title}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return "eval" if v < eval_frac else "train"


def process_article(
    title: str,
    text: str,
    max_chars: int,
    drop_short_sent: bool,
    min_sent_chars: int,
    min_alpha_ratio: float,
) -> List[str]:
    # Tokenize sentences in original text first (better boundaries), then clean.
    raw_sents = nltk.sent_tokenize(text)
    cleaned = []
    for s in raw_sents:
        s = clean_spaces(s)
        if not s:
            continue
        if drop_short_sent and not sent_ok(
            s, min_chars=min_sent_chars, min_alpha_ratio=min_alpha_ratio
        ):
            continue
        cleaned.append(s)
    if not cleaned:
        return []
    article_text = " ".join(cleaned)
    return chunk_by_chars(article_text, max_chars=max_chars)


def write_docs(out_path: str, docs: Iterable[str]):
    with open(out_path, "a", encoding="utf-8") as f:
        for d in docs:
            if d:
                f.write(d + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build train/eval text files from Wikipedia with "
        "document-level lines."
    )
    parser.add_argument(
        "--snapshot",
        default="20231101.en",
        help="Wikimedia snapshot (default: 20231101.en)",
    )
    parser.add_argument(
        "--train-out", default="data/test/wiki_train.txt", help="Train output path"
    )
    parser.add_argument(
        "--eval-out", default="data/test/wiki_eval.txt", help="Eval output path"
    )
    parser.add_argument(
        "--eval-frac", type=float, default=0.02, help="Eval fraction (default 2%)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2800,
        help="Max chars per document line (capped around ~1024 tokens)",
    )
    parser.add_argument(
        "--drop-short-sent",
        action="store_true",
        help="Drop very short/low-alpha sentences before chunking",
    )
    parser.add_argument(
        "--min-sent-chars", type=int, default=20, help="Min chars for a sentence"
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.6,
        help="Min alpha/length ratio to keep a sentence",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF streaming mode to reduce memory.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)

    ensure_punkt()

    print("Loading Wikipedia dataset...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        args.snapshot,
        split="train",
        streaming=args.streaming,
    )

    # Truncate any existing files
    open(args.train_out, "w", encoding="utf-8").close()
    open(args.eval_out, "w", encoding="utf-8").close()

    n_train = n_eval = 0
    for i, art in enumerate(ds):
        title = art.get("title") or ""
        text = art.get("text") or ""
        if not text or not title:
            continue

        bucket = split_bucket(title, args.eval_frac, args.seed)
        docs = process_article(
            title=title,
            text=text,
            max_chars=args.max_chars,
            drop_short_sent=args.drop_short_sent,
            min_sent_chars=args.min_sent_chars,
            min_alpha_ratio=args.min_alpha_ratio,
        )
        if not docs:
            continue

        if bucket == "train":
            write_docs(args.train_out, docs)
            n_train += len(docs)
        else:
            write_docs(args.eval_out, docs)
            n_eval += len(docs)

        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i+1} articles... "
                f"(train docs: {n_train}, eval docs: {n_eval})"
            )

    print(
        f"Done. Train docs: {n_train}, Eval docs: {n_eval}\n"
        f"train → {args.train_out}\n"
        f"eval  → {args.eval_out}"
    )


if __name__ == "__main__":
    main()