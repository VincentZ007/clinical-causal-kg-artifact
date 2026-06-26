#!/usr/bin/env python3
"""Parse the LLM extractor output (triples_llm.jsonl: {note_id, raw}) into
clean causal triples. Robust to markdown fences / minor JSON noise. Writes:
  - llm_triples.jsonl  : {note_id, cause, effect}  (provenance, for QA/grounding)
  - edges_sectioned_llm.tsv : cause\teffect\tfreq   (same format link_umls.py expects)
"""
import json, re, sys
from collections import Counter

IN = "triples_llm.jsonl"
OUT_TRIPLES = "llm_triples.jsonl"
OUT_EDGES = "edges_sectioned_llm.tsv"

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("input", nargs="?", default=IN)
_ap.add_argument("--out-triples", default=OUT_TRIPLES)
_ap.add_argument("--out-edges", default=OUT_EDGES)
_a = _ap.parse_args()
IN, OUT_TRIPLES, OUT_EDGES = _a.input, _a.out_triples, _a.out_edges

OBJ_RE = re.compile(r'\{\s*"cause"\s*:\s*"([^"]*)"\s*,\s*"effect"\s*:\s*"([^"]*)"\s*\}')

def parse_raw(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip().rstrip("`").strip()
    out = []
    i, j = raw.find("["), raw.rfind("]")
    if i != -1 and j > i:
        try:
            for o in json.loads(raw[i:j+1]):
                if isinstance(o, dict) and "cause" in o and "effect" in o:
                    out.append((str(o["cause"]), str(o["effect"])))
            if out:
                return out
        except Exception:
            pass
    return [(m.group(1), m.group(2)) for m in OBJ_RE.finditer(raw)]  # fallback

def normalize(span):                       # same normalization as the rule pipeline
    s = span.lower().strip()
    s = re.sub(r"[^a-z0-9%/ +-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -/+")
    return s

def ok(s):
    return 2 <= len(s) <= 80 and re.search(r"[a-z]{2,}", s)

n_notes = n_with = n_triples = n_fail = 0
edges = Counter()
with open(IN) as f, open(OUT_TRIPLES, "w") as ft:
    for line in f:
        try:
            row = json.loads(line)
        except Exception:
            n_fail += 1; continue
        n_notes += 1
        pairs = parse_raw(row.get("raw", ""))
        if pairs:
            n_with += 1
        seen = set()
        for c, e in pairs:
            cn, en = normalize(c), normalize(e)
            if not ok(cn) or not ok(en) or cn == en or (cn, en) in seen:
                continue
            seen.add((cn, en))
            n_triples += 1
            edges[(cn, en)] += 1
            ft.write(json.dumps({"note_id": row.get("note_id"), "cause": cn, "effect": en}) + "\n")

with open(OUT_EDGES, "w") as f:
    f.write("cause\teffect\tfreq\n")
    for (c, e), v in edges.most_common():
        f.write(f"{c}\t{e}\t{v}\n")

print(f"notes read           : {n_notes}  (json-parse fails: {n_fail})")
print(f"notes with >=1 triple: {n_with}  ({100*n_with/max(1,n_notes):.1f}%)")
print(f"clean triples        : {n_triples}  ({n_triples/max(1,n_notes):.2f}/note)")
print(f"unique causal edges  : {len(edges)}")
print(f"-> {OUT_TRIPLES}\n-> {OUT_EDGES}")
