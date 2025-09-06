#!/usr/bin/env python3
# build_wiki_corpus.py
import argparse
import hashlib
import io
import os
import re
from typing import Iterable, List

from datasets import load_dataset
import nltk


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def ascii_clean(text: str) -> str:
    # Lowercase, keep ASCII (replace non-ASCII with space), collapse whitespace.
    text = text.lower()
    text = "".join(ch if ord(ch) < 128 else " " for ch in text)
    # normalize whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sent_ok(s: str, min_chars: int = 20, min_alpha_ratio: float = 0.6) -> bool:
    s = s.strip()
    if len(s) < min_chars:
        return False
    alpha = sum(ch.isalpha() for ch in s)
    ratio = alpha / max(1, len(s))
    if ratio < min_alpha_ratio:
        return False
    return True


def chunk_sentences(
    sentences: List[str], max_chars: int
) -> Iterable[str]:
    """Yield chunks up to max_chars without breaking sentence order."""
    buf: List[str] = []
    cur = 0
    for s in sentences:
        if not s:
            continue
        # +1 for space when joining
        add = len(s) + (1 if buf else 0)
        if buf and cur + add > max_chars:
            yield " ".join(buf)
            buf = [s]
            cur = len(s)
        else:
            buf.append(s)
            cur += add
    if buf:
        yield " ".join(buf)


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
        s = ascii_clean(s)
        if not s:
            continue
        if drop_short_sent and not sent_ok(
            s, min_chars=min_sent_chars, min_alpha_ratio=min_alpha_ratio
        ):
            continue
        cleaned.append(s)
    if not cleaned:
        return []
    # Chunk in sentence order
    return list(chunk_sentences(cleaned, max_chars=max_chars))


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
        "--eval-frac", type=float, default=0.01, help="Eval fraction (default 1%)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Max chars per document line (keeps context; ~few KB)",
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