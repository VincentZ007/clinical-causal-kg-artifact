#!/usr/bin/env python3
"""
System 6: VC-RAG + causal screen ("vcrag_causal", ours+).

Same retrieval as VC-RAG (EHR-validated directed graph), but the retrieved edges
are RE-RANKED by the causal screen instead of by symmetric lift, and causally-robust
edges are annotated. Three tiers (recall-preserving: candidate set unchanged for
anchors with <=K neighbors; only ORDER + which-K for dense anchors changes):

  tier 0  causal PASS   (calibrated z >= Z_THRESH)         -> promote, sort by z
  tier 1  untestable    (edge not coverable by the screen) -> keep, sort by lift
  tier 2  causal FAIL   (tested but z < Z_THRESH)          -> demote (high lift but confounded)

Outputs prompts_causal.jsonl + candidates_causal.jsonl with the SAME qids as
prompts.jsonl so it slots into the existing gen/judge/eval pipeline as one more system.
"""
import csv, json, argparse, os
from collections import defaultdict

BASE = os.environ.get("ICKG_BASE", ".")
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
VAL = f"{BASE}/edges_final_llm_train.tsv"
SCREEN = f"{BASE}/causal_screen_train.json"
K = 12
Z_THRESH = 1.64

_ap = argparse.ArgumentParser()
_ap.add_argument("--mode", default="rerank", choices=["lift", "rerank", "annotate", "demote"],
                 help="lift: plain lift order, no causal layer (a base system); "
                      "rerank/annotate/demote: causal-screen variants")
_ap.add_argument("--base", default="lifttemporal", choices=["lifttemporal", "liftonly"],
                 help="lifttemporal: temporal-oriented graph (orig VC-RAG); "
                      "liftonly: LLM-extracted direction, no temporal flip (the honest base)")
_ap.add_argument("--system", default=None, help="system name")
_A = _ap.parse_args()
MODE = _A.mode
BASE_CFG = _A.base
SYSNAME = _A.system or f"vcrag_{BASE_CFG}_{MODE}"
OUT = f"{BASE}/prompts_{SYSNAME}.jsonl"
OUT_CAND = f"{BASE}/candidates_{SYSNAME}.jsonl"

SYS = ("You are a careful clinical reasoning assistant. Use the patient's problem list and any "
       "provided evidence to answer the causal question. Respond with ONLY the name of the single "
       "most likely medical condition - no explanation, no punctuation.")

cui_name = {}

# ---- causal screen scores keyed by directed edge (cause_cui, effect_cui) ----
screen = {}
for r in json.load(open(SCREEN)):
    if r.get("calibrated_z") is None:
        continue
    screen[(r["cause_cui"], r["effect_cui"])] = (r["calibrated_z"], r["adj_rr"], r["e_value"])
print(f"causal-screened directed edges: {len(screen)}")


def load_val():
    """validated directed retrieval graph (same as vcrag system 5)."""
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
            if BASE_CFG == "liftonly":
                # honest base: keep the LLM-extracted direction, NO temporal flip/expand
                cause_of[b].append((a, lift)); effect_of[a].append((b, lift))
                continue
            if d == "reversed":
                a, b = b, a
            cause_of[b].append((a, lift)); effect_of[a].append((b, lift))
            if d in ("co_coded", "bidirectional", "insufficient"):
                cause_of[a].append((b, lift)); effect_of[b].append((a, lift))
    return cause_of, effect_of


def rerank(cands, anchor, why, k=K):
    """RECALL-PRESERVING rerank: membership = VC-RAG's top-k-by-lift (identical recall),
    then reorder those k by causal tier and annotate. Only ORDER + annotation change,
    so recall(vcrag_causal) == recall(vcrag) exactly; the causal screen cannot evict the
    reference answer -- it can only change which evidence the generator sees first."""
    best = {}
    for c, w in cands:
        best[c] = max(best.get(c, 0), w)
    # 1) VC-RAG membership: top-k by lift (recall preserved in every mode)
    topk = sorted(best.items(), key=lambda kv: -kv[1])[:k]
    if MODE == "lift":                                    # plain base: no causal layer
        return [(c, lift, None) for c, lift in topk]
    rows = []   # (c, lift, info, tier) ; tier 0=pass 1=untestable 2=fail
    for c, lift in topk:
        edge = (c, anchor) if why else (anchor, c)
        sc = screen.get(edge)
        if sc and sc[0] >= Z_THRESH:
            tier, info = 0, sc
        elif sc:
            tier, info = 2, sc
        else:
            tier, info = 1, None
        rows.append((c, lift, info, tier))
    # 2) order depends on mode (topk is already lift-desc)
    if MODE == "annotate":
        ordered = rows                                   # keep lift order, just annotate
    elif MODE == "demote":
        ordered = [r for r in rows if r[3] != 2] + [r for r in rows if r[3] == 2]  # fail -> back
    else:  # rerank: tier (pass->untestable->fail), then z for pass else lift
        ordered = sorted(rows, key=lambda r: (r[3], -(r[2][0] if r[2] else r[1])))
    return [(c, lift, info) for c, lift, info, _ in ordered]


def fmt(cands):
    out = []
    for c, lift, info in cands:
        nm = cui_name.get(c, c)
        if info is not None and info[0] >= Z_THRESH:
            out.append(f"{nm} (causally robust: adj RR {info[1]:.1f}, E-value {info[2]})")
        else:
            out.append(f"{nm} (lift {lift:.1f})")
    return out


def main():
    val_cause, val_effect = load_val()
    items = [json.loads(l) for l in open(BENCH)]
    items = [x for x in items if x["llm_dir"] == "forward" and x["type"] in ("WHY", "WHATCAUSES")]
    print(f"consensus items: {len(items)}")

    out, cands_out = [], []
    n_promoted = 0
    for x in items:
        why = x["type"] == "WHY"
        anchor = x["effect_cui"] if why else x["cause_cui"]
        anchor_name = x["effect_name"] if why else x["cause_name"]
        prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
        rel = "causes" if why else "effects"
        src = val_cause if why else val_effect
        cand = rerank(src.get(anchor, []), anchor, why)
        if cand and cand[0][2] is not None and cand[0][2][0] >= Z_THRESH:
            n_promoted += 1
        lst = fmt(cand)
        if MODE == "lift":
            head = f"EHR-VALIDATED {rel} of '{anchor_name}' (co-occurrence lift in patient records): "
        else:
            head = (f"EHR-VALIDATED {rel} of '{anchor_name}', ranked by a confounder-adjusted causal "
                    f"screen (cohort design + negative-control calibration): ")
        ev = (head + ", ".join(lst) + ".\n") if lst else \
             f"No EHR-validated {rel} of '{anchor_name}' were found.\n"
        user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
                f"Answer (single condition name):")
        out.append({"qid": x["qid"], "type": x["type"], "system": SYSNAME,
                    "anchor_cui": anchor, "reference_cui": x["reference_cui"],
                    "reference_name": x["reference_name"],
                    "messages": [{"role": "system", "content": SYS},
                                 {"role": "user", "content": user}]})
        cand_cuis = [c for c, _, _ in cand]
        cands_out.append({"qid": x["qid"], "system": SYSNAME,
                          "cand_cuis": cand_cuis,
                          "ref_in_cands": x["reference_cui"] in set(cand_cuis)})
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    with open(OUT_CAND, "w") as f:
        for o in cands_out:
            f.write(json.dumps(o) + "\n")
    recall = sum(o["ref_in_cands"] for o in cands_out) / len(cands_out)
    print(f"wrote {len(out)} prompts -> {OUT}")
    print(f"questions where a causally-robust edge is ranked #1: {n_promoted}")
    print(f"retrieval recall (ref in top-{K}): {recall:.3f}")
    ex = out[0]
    print("\n=== example vcrag_causal prompt ===\n" + ex["messages"][1]["content"][:700])


if __name__ == "__main__":
    main()
