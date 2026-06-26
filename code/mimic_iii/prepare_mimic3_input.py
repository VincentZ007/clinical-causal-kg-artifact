#!/usr/bin/env python3
"""Create a deterministic MIMIC-III discharge-summary sample for the LLM extractor.

The output format matches prepare_full_input.py so the model prompt and downstream
parser stay unchanged. MIMIC-III uses upper-case NOTEEVENTS headers; this reader
also accepts lower-case equivalents.
"""
import argparse
import csv
import gzip
import json
import random
import re

csv.field_size_limit(10**9)
SECTION = re.compile(r"(?m)^[ \t]*([A-Z][A-Za-z][A-Za-z /()\\-]{2,45}):")
# Keep exactly the same section family as the established MIMIC-IV preparation script.
KEEP = ("history of present illness", "brief hospital course", "hospital course")


def opener(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def normalized_row(row):
    return {str(key).lower(): value for key, value in row.items()}


def clinical_sections(text):
    headers = list(SECTION.finditer(text or ""))
    selected = []
    for index, match in enumerate(headers):
        heading = match.group(1).strip().lower()
        if any(name in heading for name in KEEP):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            selected.append(text[match.end():end])
    return "\n".join(selected).strip()


def is_discharge(row):
    return (row.get("category") or "").strip().lower() == "discharge summary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", required=True, help="MIMIC-III NOTEEVENTS.csv[.gz]")
    parser.add_argument("--out", required=True, help="output .jsonl or .jsonl.gz")
    parser.add_argument("--limit", type=int, default=50000, help="number of notes; use 0 for all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sequential", action="store_true",
                        help="take the first usable notes; intended only for small smoke tests")
    args = parser.parse_args()

    if args.sequential:
        if args.limit <= 0:
            raise ValueError("--sequential requires a positive --limit")
        out_open = gzip.open if args.out.endswith(".gz") else open
        scanned = kept = 0
        with opener(args.notes) as source, out_open(args.out, "wt", encoding="utf-8") as target:
            for raw in csv.DictReader(source):
                row = normalized_row(raw)
                if not is_discharge(row):
                    continue
                scanned += 1
                text = clinical_sections(row.get("text", ""))
                if len(text) < 100:
                    continue
                target.write(json.dumps({"note_id": row["row_id"], "text": text[:6000]}) + "\n")
                kept += 1
                if kept >= args.limit:
                    break
        print(f"sequential discharge summaries scanned: {scanned}; usable: {kept}")
        print(f"wrote {args.out}")
        return

    # For a full-corpus run, stream once. Random selection is only needed for a
    # bounded sample, where a first pass prevents a chronological convenience sample.
    if args.limit == 0:
        out_open = gzip.open if args.out.endswith(".gz") else open
        discharge = kept = missing_sections = 0
        with opener(args.notes) as source, out_open(args.out, "wt", encoding="utf-8") as target:
            for raw in csv.DictReader(source):
                row = normalized_row(raw)
                if not is_discharge(row):
                    continue
                discharge += 1
                text = clinical_sections(row.get("text", ""))
                if len(text) < 100:
                    missing_sections += 1
                    continue
                target.write(json.dumps({"note_id": row["row_id"], "text": text[:6000]}) + "\n")
                kept += 1
        print(f"eligible discharge summaries: {discharge}; usable clinical sections: {kept}; skipped: {missing_sections}")
        print(f"wrote {args.out}")
        return

    # First pass selects note IDs only, avoiding a chronological convenience sample.
    eligible_ids = []
    with opener(args.notes) as handle:
        for raw in csv.DictReader(handle):
            row = normalized_row(raw)
            if is_discharge(row) and row.get("row_id"):
                eligible_ids.append(row["row_id"])
    if not eligible_ids:
        raise RuntimeError("No discharge summaries found. Check --notes and MIMIC access.")
    if args.limit and args.limit < len(eligible_ids):
        selected_ids = set(random.Random(args.seed).sample(eligible_ids, args.limit))
    else:
        selected_ids = set(eligible_ids)
    print(f"eligible discharge summaries: {len(eligible_ids)}; selected: {len(selected_ids)}", flush=True)

    out_open = gzip.open if args.out.endswith(".gz") else open
    seen = kept = missing_sections = 0
    with opener(args.notes) as source, out_open(args.out, "wt", encoding="utf-8") as target:
        for raw in csv.DictReader(source):
            row = normalized_row(raw)
            if row.get("row_id") not in selected_ids:
                continue
            seen += 1
            text = clinical_sections(row.get("text", ""))
            if len(text) < 100:
                missing_sections += 1
                continue
            target.write(json.dumps({"note_id": row["row_id"], "text": text[:6000]}) + "\n")
            kept += 1
    print(f"selected notes scanned: {seen}; usable clinical sections: {kept}; skipped: {missing_sections}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
