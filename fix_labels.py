#!/usr/bin/env python3
"""
Split a space/comma/semicolon separated label list into one-label-per-line .txt.

Usage:
    python fix_labels.py labels/coral_labels.txt
"""
import sys, re, pathlib

if len(sys.argv) != 2:
    sys.exit("Usage: python fix_labels.py <label_file.txt>")

p = pathlib.Path(sys.argv[1])
tokens = re.split(r"[;, \n]+", p.read_text().strip())
out = p.with_suffix(".txt")
out.write_text("\n".join(t for t in tokens if t))
print(f"✅  wrote {len(tokens)} labels to {out}")
