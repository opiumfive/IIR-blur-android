#!/usr/bin/env python3
"""Print the Markdown tables of the README from a benchmark JSONL file."""
import json
import sys
from collections import OrderedDict


def load(path):
    rows = OrderedDict()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.setdefault((r["width"], r["height"], r["sigma"], r.get("gap_ms", 0)), {})[r["variant"]] = r
    return rows


def cell(entry, key="median_ms"):
    return f"{entry[key]:.2f}" if entry else "-"


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/samsung-s21.jsonl")
    print("Rotated with the older implementations (median ms):\n")
    print("| Image | sigma | Original NEON | Original FP16 | Previous | Fast | Precise | Draft | Fast vs original | Draft vs original |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (w, h, s, gap), v in rows.items():
        if gap or "original_neon" not in v:
            continue
        o, f, d = v["original_neon"]["median_ms"], v["fast"]["median_ms"], v["draft"]["median_ms"]
        print(f"| {w}x{h} | {s:g} | {cell(v.get('original_neon'))} | {cell(v.get('original_fp16'))} | "
              f"{cell(v.get('previous'))} | {cell(v.get('fast'))} | {cell(v.get('precise'))} | "
              f"{cell(v.get('draft'))} | {o / f:.1f}x | {o / d:.1f}x |")
    print("\nBack to back, new implementation only (median ms, p90 in brackets):\n")
    print("| Image | sigma | Fast | Precise | Draft |")
    print("|---|---:|---:|---:|---:|")
    for (w, h, s, gap), v in rows.items():
        if gap or "sustained_fast" not in v:
            continue
        print(f"| {w}x{h} | {s:g} | " + " | ".join(
            f"{v[k]['median_ms']:.2f} ({v[k]['p90_ms']:.2f})" for k in ("sustained_fast", "sustained_precise", "sustained_draft")) + " |")
    print("\nIdle gap before every call, 1080x1920, sigma 30 (median ms):\n")
    print("| Gap | Original NEON | Previous | Fast | Precise | Draft |")
    print("|---:|---:|---:|---:|---:|---:|")
    for (w, h, s, gap), v in rows.items():
        if not gap:
            continue
        p = f"gap{gap}_"
        print(f"| {gap} ms | {cell(v.get(p + 'original_neon'))} | {cell(v.get(p + 'previous'))} | "
              f"{cell(v.get(p + 'fast'))} | {cell(v.get(p + 'precise'))} | {cell(v.get(p + 'draft'))} |")


if __name__ == "__main__":
    main()
