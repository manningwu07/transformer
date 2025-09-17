#!/usr/bin/env python3
import os
import argparse

def rechunk_line(line: str, max_chars: int = 2000):
    chunks = []
    start = 0
    while start < len(line):
        end = min(start + max_chars, len(line))
        if end < len(line):
            # prefer to cut at last space
            space = line.rfind(" ", start, end)
            if space != -1 and space > start + max_chars // 2:
                end = space
        chunks.append(line[start:end].strip())
        start = end
    return [c for c in chunks if c]

def rechunk_file(infile, outfile, max_chars=2000):
    with open(infile, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    out_lines = []
    for line in lines:
        out_lines.extend(rechunk_line(line, max_chars))

    with open(outfile, "w", encoding="utf-8") as f:
        for ol in out_lines:
            f.write(ol + "\n")

    print(f"✅ Rechunked {len(lines)} docs → {len(out_lines)} docs in {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    rechunk_file(args.infile, args.outfile, args.max_chars)