#!/usr/bin/env python3
"""
Build the patient-grounded causal QA benchmark from HELD-OUT (test) patients.

DUAL-SIGNAL ground truth. A causal relation is an answer key only if:
  (a) the treating clinician stated the two concepts as a cause and effect relation in the note
      (provides the held-out PATIENT grounding), AND
  (b) the edge is EHR-supported on TRAIN patients: support=='supported' & lift>=MIN_LIFT, AND
  (c) it has a DECIDED cross-admission temporal direction (forward/reversed) — and the
      CAUSE/EFFECT orientation is taken from that temporal signal, NOT from the (noisy)
      LLM/physician surface order. (co_coded / bidirectional / insufficient -> dropped:
      direction ambiguous, e.g. cardiac-arrest <-> anoxic-encephalopathy.)

The KG is built from TRAIN patients only; questions come from TEST patients. Items are
prioritized by validation strength (lift, #patients), not phrase frequency.

Types: WHY (cause of effect) | WHATCAUSES (effect of cause) | DIRECTION (A->B vs B->A).
"""
import csv, json, gzip
from collections import Counter, defaultdict
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
from link_umls import is_concept, best_candidate, JUNK_TYPES

BASE = "."
HOSP = "/path/to/MIMIC/physionet.org/files/mimiciv/3.1/hosp"
SPLIT = f"{BASE}/patient_split.tsv"
TRIPLES = f"{BASE}/llm_triples.jsonl"
ICD2CUI = f"{BASE}/icd2cui.tsv"
KG = f"{BASE}/edges_final_llm_train.tsv"
OUT = f"{BASE}/causal_qa_benchmark.jsonl"
OUT_SAMPLE = f"{BASE}/causal_qa_sample.jsonl"
P2C_CACHE = f"{BASE}/phrase2cui_test.tsv"

THRESH = 0.80
PHRASE_MINFREQ = 2
MIN_LIFT = 2.0
MAX_PROFILE = 30
PER_EDGE = 20
PER_SUBJ_TYPE = 3


def load_split():
    test = set()
    with open(SPLIT) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == "test":
                test.add(int(r["subject_id"]))
    return test


def load_kg():
    nodes, name, clean = set(), {}, {}
    with open(KG) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r["cause_cui"], r["effect_cui"]
            nodes.add(a); nodes.add(b)
            name[a] = r["cause_name"]; name[b] = r["effect_name"]
            try:
                lift = float(r["lift"]) if r["lift"] else 0.0
            except ValueError:
                lift = 0.0
            try:
                npb = int(float(r["n_pat_both"])) if r["n_pat_both"] else 0
            except ValueError:
                npb = 0
            if r["support"] == "supported" and lift >= MIN_LIFT and r["direction"] in ("forward", "reversed"):
                # orient cause->effect by the TEMPORAL signal, not the LLM surface order
                cause, effect = (a, b) if r["direction"] == "forward" else (b, a)
                key = (cause, effect)
                if key not in clean or lift > clean[key]["lift"]:
                    clean[key] = {"lift": lift, "n_pat_both": npb, "llm_dir": r["direction"]}
    return nodes, name, clean


def load_icd2cui():
    code2cui, title = {}, {}
    with open(ICD2CUI) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["cui"]:
                code2cui[(r["icd_code"], int(r["icd_version"]))] = r["cui"]
            title[r["cui"]] = r.get("long_title", "")
    return code2cui, title


def load_patient_dx(test, code2cui):
    prof = defaultdict(list); seen = defaultdict(set)
    with gzip.open(f"{HOSP}/diagnoses_icd.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            sid = int(r["subject_id"])
            if sid not in test:
                continue
            cui = code2cui.get((r["icd_code"], int(r["icd_version"])))
            if cui and cui not in seen[sid]:
                seen[sid].add(cui); prof[sid].append(cui)
    return prof


def link_phrases(phrases):
    # cache to avoid re-linking on re-runs
    cache = {}
    try:
        with open(P2C_CACHE) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                cache[r["phrase"]] = (r["cui"], r["name"])
    except FileNotFoundError:
        pass
    need = [p for p in phrases if p not in cache and is_concept(p)]
    if need:
        print(f"linking {len(need)} new test phrases ...", flush=True)
        kb = UmlsKnowledgeBase(); gen = CandidateGenerator(name="umls")
        B = 4000
        for i in range(0, len(need), B):
            chunk = need[i:i + B]
            for p, cands in zip(chunk, gen(chunk, 5)):
                best, sim = best_candidate(cands)
                if best is None or sim < THRESH:
                    cache[p] = ("", ""); continue
                ent = kb.cui_to_entity.get(best.concept_id)
                types = ent.types if ent else []
                if types and all(t in JUNK_TYPES for t in types):
                    cache[p] = ("", ""); continue
                cache[p] = (best.concept_id, ent.canonical_name if ent else p)
            print(f"  {min(i + B, len(need))}/{len(need)}", flush=True)
        with open(P2C_CACHE, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["phrase", "cui", "name"])
            for p, (c, n) in cache.items():
                w.writerow([p, c, n])
    return {p: v for p, v in cache.items() if v[0]}


def main():
    test = load_split()
    kg_nodes, kg_name, clean = load_kg()
    code2cui, title = load_icd2cui()
    print(f"test subjects: {len(test)}   kg nodes: {len(kg_nodes)}")
    print(f"CLEAN directed causal edges (supported & lift>={MIN_LIFT} & temporally decided): {len(clean)}")
    prof = load_patient_dx(test, code2cui)

    triples, pf = [], Counter()
    with open(TRIPLES) as f:
        for line in f:
            o = json.loads(line)
            try:
                sid = int(str(o["note_id"]).split("-")[0])
            except Exception:
                continue
            if sid not in test:
                continue
            triples.append((o["note_id"], sid, o["cause"], o["effect"]))
            pf[o["cause"]] += 1; pf[o["effect"]] += 1
    print(f"test triples: {len(triples)}   unique phrases: {len(pf)}")
    p2c = link_phrases([p for p, c in pf.items() if c >= PHRASE_MINFREQ])
    print(f"linked phrases: {len(p2c)}")

    def disp(cui, fb=""):
        return kg_name.get(cui) or title.get(cui) or fb or cui

    # physician statements grouped by UNORDERED concept pair (grounds the patient)
    pair_pat = defaultdict(list)
    for nid, sid, c, e in triples:
        if c not in p2c or e not in p2c:
            continue
        cc, ce = p2c[c][0], p2c[e][0]
        if cc != ce:
            pair_pat[frozenset((cc, ce))].append((nid, sid))

    def profile_of(sid):
        cuis = prof.get(sid, [])[:MAX_PROFILE]
        return set(cuis), [{"cui": x, "name": disp(x)} for x in cuis]

    cand = {"WHY": [], "WHATCAUSES": [], "DIRECTION": []}
    subj_n = Counter()
    for (tc, te) in sorted(clean, key=lambda k: (-clean[k]["lift"], -clean[k]["n_pat_both"])):
        attr = clean[(tc, te)]
        cn, en = disp(tc), disp(te)
        seen_sid = set()
        for nid, sid in pair_pat.get(frozenset((tc, te)), []):
            if sid in seen_sid:
                continue
            seen_sid.add(sid)
            if len(seen_sid) > PER_EDGE:
                break
            pset, pnamed = profile_of(sid)
            base = dict(subject_id=sid, note_id=nid, cause_cui=tc, cause_name=cn,
                        effect_cui=te, effect_name=en, patient_profile=pnamed,
                        lift=attr["lift"], n_pat_both=attr["n_pat_both"], llm_dir=attr["llm_dir"],
                        cause_in_profile=tc in pset, effect_in_profile=te in pset)
            if subj_n[(sid, "WHY")] < PER_SUBJ_TYPE:
                subj_n[(sid, "WHY")] += 1
                cand["WHY"].append({**base, "type": "WHY", "reference_cui": tc, "reference_name": cn,
                    "question": f"This patient's problem list is given. What is the single most likely underlying CAUSE of their '{en}'?"})
            if subj_n[(sid, "WC")] < PER_SUBJ_TYPE:
                subj_n[(sid, "WC")] += 1
                cand["WHATCAUSES"].append({**base, "type": "WHATCAUSES", "reference_cui": te, "reference_name": en,
                    "question": f"In a patient like this, what condition is '{cn}' most likely to CAUSE or lead to?"})
            if subj_n[(sid, "DIR")] < PER_SUBJ_TYPE:
                subj_n[(sid, "DIR")] += 1
                flip = (sid % 2 == 1)   # vary option order to avoid position bias
                left, right = (te, tc) if flip else (tc, te)
                correct = "B" if flip else "A"
                cand["DIRECTION"].append({**base, "type": "DIRECTION",
                    "reference_direction": "forward", "reference_name": f"{cn} -> {en}",
                    "correct_option": correct,
                    "question": (f"Which causal direction is correct?  A) '{disp(left)}' -> '{disp(right)}'   "
                                 f"B) '{disp(right)}' -> '{disp(left)}'")})

    items, qid = [], 0
    for typ, lst in cand.items():
        for x in lst:
            x["qid"] = f"q{qid:06d}"; qid += 1
            items.append(x)
        print(f"{typ:11s} {len(lst)}")
    with open(OUT, "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    sample = []
    for typ in cand:
        sample += [x for x in items if x["type"] == typ][:40]
    with open(OUT_SAMPLE, "w") as f:
        for x in sample:
            f.write(json.dumps(x) + "\n")

    print(f"\ntotal items: {len(items)}  ({OUT})")
    print(f"human-check sample: {len(sample)}  ({OUT_SAMPLE})")
    cinp = Counter(x["type"] for x in items if x.get("cause_in_profile"))
    tc_ = Counter(x["type"] for x in items)
    print("\ntype         total  cause_in_profile(easy)")
    for t in ["WHY", "WHATCAUSES", "DIRECTION"]:
        print(f"{t:11s} {tc_[t]:6d}  {cinp[t]:6d}")


if __name__ == "__main__":
    main()
