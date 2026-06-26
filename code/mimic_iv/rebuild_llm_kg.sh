#!/usr/bin/env bash
# Rebuild the causal KG from the LLM-extracted triples and compare against the
# rule version. Run AFTER fetch_results.sh has pulled triples_llm.jsonl.
# All outputs use the _llm suffix, so the rule-version files are left intact.
set -e
cd /media/lansu/Expansion/PHD/causal-kg
R="conda run -n causal-kg python"

echo "=== [1/5] parse LLM raw -> triples + edges_sectioned_llm.tsv ==="
$R parse_llm_triples.py

echo "=== [2/5] UMLS entity linking (LLM) -> edges_cui_llm.tsv ==="
$R link_umls.py --edges edges_sectioned_llm.tsv --map-out phrase2cui_llm.tsv --cui-edges edges_cui_llm.tsv

echo "=== [3/5] structured effect-support (LLM) -> edges_cui_validated_llm.tsv ==="
$R validate_edges.py --kg edges_cui_llm.tsv --out edges_cui_validated_llm.tsv

echo "=== [4/5] cross-admission temporal direction (LLM) -> edges_final_llm.tsv ==="
$R temporal_direction.py --kg edges_cui_validated_llm.tsv --out edges_final_llm.tsv

echo "=== [5/5] RULE vs LLM comparison ==="
$R compare_rule_vs_llm.py

echo "=== DONE: edges_cui_llm.tsv / edges_final_llm.tsv / llm_triples.jsonl ==="
