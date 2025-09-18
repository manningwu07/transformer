#!/usr/bin/env python3
import argparse
import os

def slice_text(text: str, max_chars=1200, min_tail=300):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # cut at the last space before end
            space = text.rfind(" ", start, end)
            if space != -1 and space > start:
                end = space
        chunks.append(text[start:end].strip())
        start = end

    # handle "tiny tail" merge
    out = []
    for i, ch in enumerate(chunks):
        if i < len(chunks) - 1 and len(chunks[i+1]) < min_tail:
            # merge this and next
            chunks[i+1] = ch + " " + chunks[i+1]
        else:
            out.append(ch)
    return [c for c in out if c]

def reslice_file(infile, outfile, max_chars=1200, min_tail=300):
    with open(infile, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    out_lines = []
    for line in lines:
        out_lines.extend(slice_text(line, max_chars=max_chars, min_tail=min_tail))

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for ol in out_lines:
            f.write(ol + "\n")

    print(f"✅ Resliced {len(lines)} orig lines → {len(out_lines)} new lines in {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--min-tail", type=int, default=300)
    args = ap.parse_args()
    reslice_file(args.infile, args.outfile, args.max_chars, args.min_tail)