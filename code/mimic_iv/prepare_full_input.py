#!/usr/bin/env python3
"""Build the FULL LLM-extractor input: clinical sections for every discharge note,
gzip-compressed for transfer to the GPU node."""
import gzip, csv, json, re
csv.field_size_limit(10**9)
NOTE = "/media/lansu/Expansion/PHD/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
OUT = "/media/lansu/Expansion/PHD/causal-kg/full_input.jsonl.gz"
SEC = re.compile(r"(?m)^[ \t]*([A-Z][A-Za-z][A-Za-z /()\-]{2,45}):")
KEEP = ("history of present illness", "brief hospital course", "hospital course")

def sections(text):
    heads = list(SEC.finditer(text)); out = []
    for i, m in enumerate(heads):
        if any(k in m.group(1).strip().lower() for k in KEEP):
            end = heads[i+1].start() if i+1 < len(heads) else len(text)
            out.append(text[m.end():end])
    return "\n".join(out)

c = kept = 0
with gzip.open(NOTE, "rt") as f, gzip.open(OUT, "wt") as o:
    for row in csv.DictReader(f):
        c += 1
        sec = sections(row["text"]).strip()
        if len(sec) < 100:
            continue
        o.write(json.dumps({"note_id": row["note_id"], "text": sec[:6000]}) + "\n")
        kept += 1
print(f"scanned {c} notes, wrote {kept} with usable clinical sections -> {OUT}")
