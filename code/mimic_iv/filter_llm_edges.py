import csv, sys
MINFREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2
n = k = 0
with open("edges_sectioned_llm.tsv") as f, open("edges_sectioned_llm_f2.tsv", "w", newline="") as g:
    rd = csv.reader(f, delimiter="\t"); w = csv.writer(g, delimiter="\t")
    w.writerow(next(rd))
    for c, e, fr in rd:
        n += 1
        if int(fr) >= MINFREQ:
            w.writerow([c, e, fr]); k += 1
print(f"kept {k}/{n} edges (freq>={MINFREQ}) -> edges_sectioned_llm_f2.tsv")
