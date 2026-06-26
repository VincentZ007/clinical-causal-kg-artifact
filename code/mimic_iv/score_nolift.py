#!/usr/bin/env python3
"""
Score B's lift-free stratified QA. Judge-free first read: deterministic lenient
name-match of the predicted answer against the reference concept, reported PER STRATUM
(H / L_weak / L_nottestable) PER SYSTEM, alongside candidate-recall. A Qwen-judge
EXACT/ALT/WRONG pass (build_nolift_judge.py) is the robustness follow-up.

Inputs : answers_nolift.jsonl ({qid,system,type,answer}), candidates_nolift.jsonl, causal_qa_nolift_sample.jsonl
Output : nolift_scores.json + stdout (per-stratum correctness + recall + McNemar vcrag vs unvalidated)
"""
import json, re
from collections import defaultdict, Counter
from math import comb

BASE = "/media/lansu/Expansion/PHD/causal-kg"
ANS = f"{BASE}/answers_nolift.jsonl"
CAND = f"{BASE}/candidates_nolift.jsonl"
BENCH = f"{BASE}/causal_qa_nolift_sample.jsonl"
STRATA = ["H", "L_weak", "L_nottestable"]
SYSTEMS = ["closed", "assoc", "unvalidated", "vcrag"]


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def match(ans, ref):
    a, r = norm(ans), norm(ref)
    if not a or not r:
        return False
    if a == r:
        return True
    # lenient containment (guard against trivial short tokens)
    if len(r) >= 5 and r in a:
        return True
    if len(a) >= 5 and a in r:
        return True
    # strong token overlap (Jaccard over content words)
    aw, rw = set(a.split()), set(r.split())
    if rw and len(aw & rw) / len(rw) >= 0.6:
        return True
    return False


def main():
    meta = {}
    for l in open(BENCH):
        x = json.loads(l)
        st = "H" if x["stratum"] == "H" else x["lsub"]
        meta[x["qid"]] = {"ref": x["reference_name"], "stratum": st}
    recall = {}
    for l in open(CAND):
        c = json.loads(l)
        recall[(c["qid"], c["system"])] = c["ref_in_cands"]

    correct = defaultdict(lambda: Counter())   # (system) -> {stratum: n_correct}
    total = defaultdict(lambda: Counter())
    rec = defaultdict(lambda: Counter())
    per_item = defaultdict(dict)                # (qid) -> {system: correct?}
    for l in open(ANS):
        a = json.loads(l)
        m = meta.get(a["qid"])
        if not m:
            continue
        st = m["stratum"]
        ok = match(a["answer"], m["ref"])
        correct[a["system"]][st] += ok
        total[a["system"]][st] += 1
        rec[a["system"]][st] += recall.get((a["qid"], a["system"]), False)
        per_item[a["qid"]][a["system"]] = ok

    print("PER-STRATUM correctness (lenient name-match) | candidate-recall:")
    print(f"{'system':>12} | " + " | ".join(f"{s:>22}" for s in STRATA))
    out = {"correctness": {}, "recall": {}}
    for s in SYSTEMS:
        cells = []
        for st in STRATA:
            n = total[s][st]
            if n:
                cells.append(f"{100*correct[s][st]/n:5.1f}% rec {100*rec[s][st]/n:4.0f}% (n{n})")
            else:
                cells.append("-")
        print(f"{s:>12} | " + " | ".join(f"{c:>22}" for c in cells))
        out["correctness"][s] = {st: (correct[s][st] / total[s][st] if total[s][st] else None) for st in STRATA}
        out["recall"][s] = {st: (rec[s][st] / total[s][st] if total[s][st] else None) for st in STRATA}

    # McNemar vcrag vs unvalidated per stratum (paired on qid)
    print("\nMcNemar vcrag vs unvalidated (paired), per stratum:")
    for st in STRATA:
        b = c = 0
        for qid, d in per_item.items():
            if meta.get(qid, {}).get("stratum") != st:
                continue
            if "vcrag" in d and "unvalidated" in d:
                if d["vcrag"] and not d["unvalidated"]:
                    b += 1
                elif d["unvalidated"] and not d["vcrag"]:
                    c += 1
        m = b + c
        p = min(1.0, sum(comb(m, k) for k in range(min(b, c) + 1)) * 2 / (2 ** m)) if m else 1.0
        print(f"  {st:>14}: vcrag-only-right {b}, unval-only-right {c}, p={p:.2e}")
        out.setdefault("mcnemar", {})[st] = {"vcrag_only": b, "unval_only": c, "p": p}

    with open(f"{BASE}/nolift_scores.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {BASE}/nolift_scores.json")


if __name__ == "__main__":
    main()
