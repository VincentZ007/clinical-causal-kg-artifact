#!/usr/bin/env python3
"""Build a compact pilot input for the LLM extractor: clinical sections per note."""
import gzip, csv, json, re, sys
csv.field_size_limit(10**9)
NOTE = "/media/lansu/Expansion/PHD/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
OUT = "/media/lansu/Expansion/PHD/causal-kg/pilot_input.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEC = re.compile(r"(?m)^[ \t]*([A-Z][A-Za-z][A-Za-z /()\-]{2,45}):")
KEEP = ("history of present illness", "brief hospital course", "hospital course")

def sections(text):
    heads = list(SEC.finditer(text)); out = []
    for i, m in enumerate(heads):
        if any(k in m.group(1).strip().lower() for k in KEEP):
            end = heads[i+1].start() if i+1 < len(heads) else len(text)
            out.append(text[m.end():end])
    return "\n".join(out)

c = 0
with gzip.open(NOTE, "rt") as f, open(OUT, "w") as o:
    for row in csv.DictReader(f):
        sec = sections(row["text"]).strip()
        if len(sec) < 100:
            continue
        o.write(json.dumps({"note_id": row["note_id"], "text": sec[:6000]}) + "\n")
        c += 1
        if c >= N:
            break
print(f"wrote {c} notes -> {OUT}")
