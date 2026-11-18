#!/usr/bin/env python3
"""Multiset difference of (opcode, result shape) between the ENTRY computations of two HLO text dumps.

Robust to instruction reordering and renaming (unlike a line diff). Read-only.
    entry_opdiff.py A.txt B.txt
"""
import collections
import re
import sys

INSTR_RE = re.compile(
    r"^\s*(ROOT\s+)?%(?P<name>[A-Za-z0-9_.\-]+)\s*=\s*(?P<shape>.+?)\s+(?P<op>[a-z][a-z0-9\-]*)\((?P<args>.*?)\)(?P<rest>.*)$")


def strip_layout(shape):
    def repl(m):
        s = re.search(r"S\((\d+)\)", m.group(0))
        return "@S" + s.group(1) if s else ""
    return re.sub(r"\{[^{}]*\}", repl, shape)


def entry_ops(path):
    c = collections.Counter()
    entry = False
    n = 0
    with open(path, errors="replace") as f:
        for line in f:
            if not entry:
                entry = line.startswith("ENTRY")
                continue
            if line.strip() == "}":
                break
            m = INSTR_RE.match(line)
            if m:
                n += 1
                op = m.group("op")
                extra = ""
                t = re.search(r'custom_call_target="([^"]+)"', m.group("rest"))
                if t:
                    extra = " target=" + t.group(1)
                c[(op + extra, strip_layout(m.group("shape")))] += 1
    return c, n


a, na = entry_ops(sys.argv[1])
b, nb = entry_ops(sys.argv[2])
print(f"# A ENTRY instructions: {na}; B: {nb}; B - A = {nb - na}")
keys = sorted(set(a) | set(b), key=lambda k: (-(abs(b[k] - a[k])), k))
for k in keys:
    if a[k] != b[k]:
        print(f"{b[k] - a[k]:+5d}  (A {a[k]:4d}, B {b[k]:4d})  {k[0]}  {k[1]}")
