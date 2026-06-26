#!/usr/bin/env python3
"""
Build the controlled VC-RAG comparison prompts. Same generator + same question;
only the retrieved KNOWLEDGE differs across 4 systems:

  closed       : no retrieval (parametric only)                         [floor]
  assoc        : UNDIRECTED co-occurrence neighbours (lift, no direction)
  unvalidated  : RAW LLM directed causal graph (direction, NO EHR validation)  [CausalRAG-style]
  vcrag        : EHR-VALIDATED directed causal graph (supported lift + temporal dir)  [ours]

WHY  -> retrieve CAUSES of the effect (in-edges).  WHATCAUSES -> retrieve EFFECTS (out-edges).
Runs on the consensus subset (llm_dir==forward) so the ground truth is trustworthy.
Output: prompts.jsonl  {qid,type,system,anchor_cui,reference_cui,reference_name,messages}
"""
import csv, json
from collections import defaultdict

BASE = "/media/lansu/Expansion/PHD/causal-kg"
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
RAW = f"{BASE}/edges_cui_llm_train.tsv"            # unvalidated directed
ASSOC = f"{BASE}/edges_cui_validated_llm_train.tsv"  # lift (symmetric)
VAL = f"{BASE}/edges_final_llm_train.tsv"            # validated + temporal direction
OUT = f"{BASE}/prompts.jsonl"
OUT_CAND = f"{BASE}/candidates.jsonl"
K = 12

SYS = ("You are a careful clinical reasoning assistant. Use the patient's problem list and any "
       "provided evidence to answer the causal question. Respond with ONLY the name of the single "
       "most likely medical condition - no explanation, no punctuation.")

cui_name = {}


def load_raw():
    cause_of, effect_of = defaultdict(list), defaultdict(list)
    with open(RAW) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cui_name[r["cause_cui"]] = r["cause_name"]; cui_name[r["effect_cui"]] = r["effect_name"]
            w = float(r["freq"])
            cause_of[r["effect_cui"]].append((r["cause_cui"], w))   # causes of effect
            effect_of[r["cause_cui"]].append((r["effect_cui"], w))  # effects of cause
    return cause_of, effect_of


def load_assoc():
    neigh = defaultdict(dict)
    with open(ASSOC) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cui_name[r["cause_cui"]] = r["cause_name"]; cui_name[r["effect_cui"]] = r["effect_name"]
            try:
                lift = float(r["lift"])
            except ValueError:
                continue
            a, b = r["cause_cui"], r["effect_cui"]
            neigh[a][b] = max(neigh[a].get(b, 0), lift)
            neigh[b][a] = max(neigh[b].get(a, 0), lift)
    return {k: list(v.items()) for k, v in neigh.items()}


def load_val():
    """validated directed: supported edges, temporally oriented, reversed excluded."""
    cause_of, effect_of = defaultdict(list), defaultdict(list)
    with open(VAL) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cui_name[r["cause_cui"]] = r["cause_name"]; cui_name[r["effect_cui"]] = r["effect_name"]
            if r["support"] != "supported":
                continue
            try:
                lift = float(r["lift"])
            except ValueError:
                lift = 0.0
            a, b, d = r["cause_cui"], r["effect_cui"], r["direction"]
            if d == "reversed":
                a, b = b, a               # temporal says effect precedes -> flip
            elif d == "forward":
                pass
            # forward/reversed -> directed; co_coded/bidirectional/insufficient -> both ways
            cause_of[b].append((a, lift)); effect_of[a].append((b, lift))
            if d in ("co_coded", "bidirectional", "insufficient"):
                cause_of[a].append((b, lift)); effect_of[b].append((a, lift))
    return cause_of, effect_of


def topk(cands, k=K):
    best = {}
    for c, w in cands:
        best[c] = max(best.get(c, 0), w)
    return sorted(best.items(), key=lambda kv: -kv[1])[:k]


def names(cands, with_lift=False):
    out = []
    for c, w in cands:
        nm = cui_name.get(c, c)
        out.append(f"{nm} (lift {w:.1f})" if with_lift else nm)
    return out


def main():
    raw_cause, raw_effect = load_raw()
    assoc = load_assoc()
    val_cause, val_effect = load_val()
    items = [json.loads(l) for l in open(BENCH)]
    items = [x for x in items if x["llm_dir"] == "forward" and x["type"] in ("WHY", "WHATCAUSES")]
    print(f"consensus items: {len(items)}")

    out, cands_out = [], []
    for x in items:
        why = x["type"] == "WHY"
        anchor = x["effect_cui"] if why else x["cause_cui"]
        anchor_name = x["effect_name"] if why else x["cause_name"]
        prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
        rel = "causes" if why else "effects"
        for system in ("closed", "assoc", "unvalidated", "vcrag"):
            cand = []
            if system == "closed":
                ev = ""
            elif system == "assoc":
                cand = topk(assoc.get(anchor, []))
                ev = f"Conditions statistically co-occurring with '{anchor_name}' in similar patients: " \
                     + ", ".join(names(cand)) + ".\n"
            elif system == "unvalidated":
                src = raw_cause if why else raw_effect
                cand = topk(src.get(anchor, []))
                ev = f"Candidate {rel} of '{anchor_name}' extracted from clinical notes: " \
                     + ", ".join(names(cand)) + ".\n"
            else:  # vcrag
                src = val_cause if why else val_effect
                cand = topk(src.get(anchor, []))
                lst = names(cand, with_lift=True)
                ev = (f"EHR-VALIDATED {rel} of '{anchor_name}' (each confirmed by co-occurrence lift "
                      f"AND temporal precedence in patient records): " + ", ".join(lst) + ".\n") if lst else \
                     f"No EHR-validated {rel} of '{anchor_name}' were found.\n"
            user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
                    f"Answer (single condition name):")
            out.append({"qid": x["qid"], "type": x["type"], "system": system,
                        "anchor_cui": anchor, "reference_cui": x["reference_cui"],
                        "reference_name": x["reference_name"],
                        "messages": [{"role": "system", "content": SYS},
                                     {"role": "user", "content": user}]})
            cands_out.append({"qid": x["qid"], "system": system,
                              "cand_cuis": [c for c, _ in cand],
                              "ref_in_cands": x["reference_cui"] in {c for c, _ in cand}})
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    with open(OUT_CAND, "w") as f:
        for o in cands_out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} prompts ({len(items)} items x 4 systems) -> {OUT}")
    print(f"wrote candidates -> {OUT_CAND}")
    # show one full example
    ex = [o for o in out if o["system"] == "vcrag"][0]
    print("\n=== example VC-RAG prompt ===\n" + ex["messages"][1]["content"])


if __name__ == "__main__":
    main()
