import csv
from collections import Counter
freq = Counter(); n = 0; ef = Counter()
with open("edges_sectioned_llm.tsv") as f:
    rd = csv.reader(f, delimiter="\t"); next(rd)
    for c, e, fr in rd:
        fr = int(fr); n += 1; ef[fr] += 1
        freq[c] += fr; freq[e] += fr
ph = len(freq)
print("unique edges:", n)
print("unique phrases:", ph)
for thr in (1, 2, 3, 5, 10):
    print(f"  phrases total-freq >= {thr}: {sum(1 for v in freq.values() if v >= thr)}")
print("edges freq==1:", ef.get(1, 0), f"({100*ef.get(1,0)/n:.1f}%)")
for thr in (2, 3, 5, 10):
    print(f"edges freq>={thr}:", sum(v for k, v in ef.items() if k >= thr))
