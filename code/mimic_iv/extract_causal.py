#!/usr/bin/env python3
"""
Dependency-free rule-based causal-triple extractor for MIMIC-IV-Note discharge summaries.

This is the rule baseline for a causal-KG paper: it scans for explicit causal
cue phrases, parses the left/right clause around each cue, and emits directed
(cause, CAUSES, effect) triples. Direction is cue-specific.

No external deps (stdlib only) so it runs on CPU / Python 3.13 with nothing installed.
Later this gets compared against an LLM extractor + UMLS entity linking.
"""
import gzip, csv, re, sys, json, argparse
from collections import Counter

csv.field_size_limit(10**9)

# cue phrase -> (left_role, right_role). "effect" is the consequence, "cause" the antecedent.
#   "X due to Y"        -> effect=X, cause=Y
#   "X complicated by Y"-> cause=X, effect=Y   (X is the condition, Y the resulting complication)
#   "X resulting in Y"  -> cause=X, effect=Y
CUES = {
    "due to":          ("effect", "cause"),
    "secondary to":    ("effect", "cause"),
    "caused by":       ("effect", "cause"),
    "attributed to":   ("effect", "cause"),
    "because of":      ("effect", "cause"),
    "as a result of":  ("effect", "cause"),
    "complicated by":  ("cause",  "effect"),
    "resulting in":    ("cause",  "effect"),
    "resulted in":     ("cause",  "effect"),
    "leading to":      ("cause",  "effect"),
    "led to":          ("cause",  "effect"),
}
CUE_RE = re.compile(r"(?i)\b(" + "|".join(re.escape(c) for c in CUES) + r")\b")

# split a side-string at strong clause boundaries (NOT at of/in/with -> keep medical phrases intact)
BOUNDARY_RE = re.compile(r"(?i)[,;:]|\b(?:and|but|which|who|that|however|although|while|whereas|then)\b")
SENT_RE = re.compile(r"(?<=[.;])\s+|\n+")

# section-aware extraction: only mine clinical-reasoning sections, skip patient-facing boilerplate
SEC_HEADER = re.compile(r"(?m)^[ \t]*([A-Z][A-Za-z][A-Za-z /()\-]{2,45}):")
KEEP_SECTIONS = ("history of present illness", "brief hospital course", "hospital course",
                 "past medical history", "assessment and plan", "assessment")
SECOND_PERSON = re.compile(r"(?i)\b(you|your|yourself)\b")

def clinical_sections(text):
    """Yield bodies of whitelisted clinical sections only."""
    heads = list(SEC_HEADER.finditer(text))
    if not heads:
        yield text
        return
    for i, m in enumerate(heads):
        name = m.group(1).strip().lower()
        if not any(k in name for k in KEEP_SECTIONS):
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        yield text[m.end():end]

LEAD_STRIP = {"the","a","an","this","that","these","those","his","her","their","its",
              "patient's","pt's","likely","most","presumed","possible","probable",
              "possibly","presumably","known","some","any","no"}
NEG_LEAD = {"no","denies","without","negative","ruled"}
STOP = {"the","a","an","of","to","in","on","with","for","and","or","is","was","be",
        "as","at","by","from","her","his","their","this","that","patient","pt"}

def clean_span(s, take_first):
    """take_first=True -> keep clause nearest *after* cue (first chunk);
       take_first=False -> keep clause nearest *before* cue (last chunk)."""
    parts = [p.strip() for p in BOUNDARY_RE.split(s) if p and p.strip()]
    if not parts:
        return ""
    chunk = parts[0] if take_first else parts[-1]
    toks = chunk.split()
    # trim leading filler/determiners
    while toks and toks[0].lower().strip(".,") in LEAD_STRIP:
        toks = toks[1:]
    # cap length, drop de-id artifacts and pure punctuation
    toks = [t for t in toks if t.strip("_") != "" ]
    toks = toks[:10] if take_first else toks[-10:]
    span = " ".join(toks).strip(" .,:;-")
    span = re.sub(r"\s+", " ", span)
    return span

def has_content(span):
    if len(span) < 3 or len(span) > 70:
        return False
    words = re.findall(r"[A-Za-z]{3,}", span)
    return any(w.lower() not in STOP for w in words)

def is_negated(effect_raw):
    head = effect_raw.lower().split()[:2]
    return any(w in NEG_LEAD for w in head)

def extract(text, sectioned=False, drop_second_person=False):
    triples = []
    chunks = clinical_sections(text) if sectioned else [text]
    for chunk in chunks:
      for sent in SENT_RE.split(chunk):
        sent = sent.strip()
        if not sent:
            continue
        if drop_second_person and SECOND_PERSON.search(sent):
            continue
        for m in CUE_RE.finditer(sent):
            cue = m.group(1).lower()
            left_role, right_role = CUES[cue]
            left = sent[:m.start()]
            right = sent[m.end():]
            left_span = clean_span(left, take_first=False)
            right_span = clean_span(right, take_first=True)
            roles = {left_role: left_span, right_role: right_span}
            cause, effect = roles["cause"], roles["effect"]
            if not (has_content(cause) and has_content(effect)):
                continue
            neg = is_negated(effect if left_role == "effect" else right)
            triples.append({
                "cause": cause, "effect": effect, "cue": cue,
                "predicate": "CAUSES", "negated": neg,
                "sentence": re.sub(r"\s+", " ", sent)[:240],
            })
    return triples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default="/path/to/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz")
    ap.add_argument("--n", type=int, default=1500, help="number of notes to sample")
    ap.add_argument("--out", default="./triples_sample.jsonl")
    args = ap.parse_args()

    n = 0
    all_triples = []
    with gzip.open(args.gz, "rt") as f:
        for row in csv.DictReader(f):
            all_triples.extend(extract(row["text"]))
            n += 1
            if n >= args.n:
                break

    clean = [t for t in all_triples if not t["negated"]]
    with open(args.out, "w") as f:
        for t in clean:
            f.write(json.dumps(t) + "\n")

    cue_ct = Counter(t["cue"] for t in all_triples)
    cause_ct = Counter(t["cause"].lower() for t in clean)
    effect_ct = Counter(t["effect"].lower() for t in clean)

    print(f"notes sampled        : {n}")
    print(f"raw triples          : {len(all_triples)}  ({len(all_triples)/n:.2f}/note)")
    print(f"negated (dropped)    : {sum(t['negated'] for t in all_triples)}")
    print(f"clean triples        : {len(clean)}  ({len(clean)/n:.2f}/note)")
    print(f"unique cause phrases : {len(cause_ct)}")
    print(f"unique effect phrases: {len(effect_ct)}")
    print(f"-> written to {args.out}")

    print("\n== triples per cue ==")
    for c, v in cue_ct.most_common():
        print(f"  {c:18s} {v}")
    print("\n== top 15 effect phrases ==")
    for c, v in effect_ct.most_common(15):
        print(f"  {v:4d}  {c}")
    print("\n== top 15 cause phrases ==")
    for c, v in cause_ct.most_common(15):
        print(f"  {v:4d}  {c}")

    print("\n== 25 example triples (every Nth) ==")
    step = max(1, len(clean)//25)
    for t in clean[::step][:25]:
        print(f"  [{t['cue']}]  ({t['cause']})  --CAUSES-->  ({t['effect']})")

if __name__ == "__main__":
    main()
