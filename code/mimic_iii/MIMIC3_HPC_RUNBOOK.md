# MIMIC-III external-audit runbook

## Goal and scope

This run repeats the extraction and validation audit on a deterministic random sample
of 50,000 MIMIC-III discharge summaries. It tests whether the main audit finding
transfers across a second EHR release:

- lift is an existence-support signal;
- diagnosis coding order is not a dependable direction signal; and
- SemMedDB is an independent but coverage-limited direction reference.

It is **not** an external causal-QA experiment. Do not claim cross-dataset QA
generalization unless a separate MIMIC-III benchmark is constructed and evaluated.

## 1. Connect and create a project directory

In MobaXterm, open an SSH session to the school HPC login host. In the terminal run:

```bash
mkdir -p $HOME/projects/ickg-mimic3-audit
cd $HOME/projects/ickg-mimic3-audit
pwd
```

The last line must point to a folder under your own home. Do not work in another
student's directory.

## 2. Upload code only

Use MobaXterm's left SFTP panel to upload these repository files to that directory:

```text
prepare_mimic3_input.py
extract_llm_full.py
parse_llm_triples.py
link_umls.py
validate_edges.py
temporal_direction.py
semmeddb_ground_truth.py
direction_sensitivity.py
summarize_audit.py
extract_full.sbatch
```

Check the upload:

```bash
for f in prepare_mimic3_input.py extract_llm_full.py parse_llm_triples.py link_umls.py \
  validate_edges.py temporal_direction.py semmeddb_ground_truth.py direction_sensitivity.py \
  summarize_audit.py extract_full.sbatch
do
  test -s "$f" && echo "OK   $f" || echo "MISS $f"
done
```

Stop if any file says `MISS`.

## 3. Verify access and software before a GPU job

MIMIC-III and SemMedDB must already be in a credentialed HPC location. Do not upload
patient notes to a personal machine or public storage. Set the three real paths here:

```bash
export MIMIC3_NOTES='/absolute/path/to/NOTEEVENTS.csv.gz'
export MIMIC3_HOSP='/absolute/path/to/mimic-iii-clinical-database-1.4'
export SEMMED='/absolute/path/to/semmedVER43_R_PREDICATION.csv.gz'
source /sw/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate test
test -r "$MIMIC3_NOTES" && echo 'notes OK'
test -r "$MIMIC3_HOSP/ADMISSIONS.csv.gz" && echo 'admissions OK'
test -r "$MIMIC3_HOSP/DIAGNOSES_ICD.csv.gz" && echo 'diagnoses OK'
test -r "$MIMIC3_HOSP/D_ICD_DIAGNOSES.csv.gz" && echo 'dictionary OK'
test -r "$SEMMED" && echo 'SemMedDB OK'
python -c "import pandas, scispacy; print('Python packages OK')"
```

All six checks must succeed. Confirm with the HPC administrator that the AWQ Qwen model
is available on GPU nodes before attempting the 50k run.

## 4. Prepare a 200-note smoke sample

```bash
mkdir -p smoke_200 run_50000
python prepare_mimic3_input.py --notes "$MIMIC3_NOTES" \
  --out smoke_200/input.jsonl.gz --limit 200 --seed 2026
gzip -cd smoke_200/input.jsonl.gz | wc -l
gzip -cd smoke_200/input.jsonl.gz | head -1
```

The output must have a non-zero line count and start with `note_id`. The text field
must contain a clinical section, not a blank string.

## 5. Run the GPU smoke test

```bash
INPUT=smoke_200/input.jsonl.gz OUTPUT=smoke_200/triples_llm.jsonl sbatch extract_full.sbatch
squeue -u "$USER"
```

After the job finishes:

```bash
test -s smoke_200/triples_llm.jsonl && echo 'extractor output OK'
wc -l smoke_200/triples_llm.jsonl
tail -20 causal-llm-*.out
```

The extractor output count should match the input count and the log should end in
`ALL DONE`. If it reports out-of-memory, set `gpu_memory_utilization=0.85` in
`extract_llm_full.py` and repeat only this 200-note test.

## 6. Build the deterministic 50k sample

```bash
python prepare_mimic3_input.py --notes "$MIMIC3_NOTES" \
  --out run_50000/input.jsonl.gz --limit 50000 --seed 2026
gzip -cd run_50000/input.jsonl.gz | wc -l
```

Record the eligible, selected, usable, and skipped counts printed by the script. The
fixed seed is part of the reproducibility record.

## 7. Extract causal triples

```bash
INPUT=run_50000/input.jsonl.gz OUTPUT=run_50000/triples_llm.jsonl sbatch extract_full.sbatch
squeue -u "$USER"
```

The extractor is resumable. If the job times out, submit the identical command again;
it skips `note_id`s already in `triples_llm.jsonl`. Do not delete a partial output.

After completion:

```bash
gzip -cd run_50000/input.jsonl.gz | wc -l
wc -l run_50000/triples_llm.jsonl
tail -20 causal-llm-*.out
```

The counts must match. Otherwise resubmit once and inspect the log before proceeding.

## 8. Parse, link, and run the EHR audit

Run these where UMLS/scispaCy is available. If login-node CPU work is restricted, submit
them through your cluster's CPU queue instead.

```bash
python parse_llm_triples.py run_50000/triples_llm.jsonl \
  --out-triples run_50000/llm_triples.jsonl \
  --out-edges run_50000/edges_sectioned_llm.tsv

python link_umls.py --edges run_50000/edges_sectioned_llm.tsv \
  --map-out run_50000/phrase2cui.tsv \
  --cui-edges run_50000/edges_cui.tsv

python validate_edges.py --hosp "$MIMIC3_HOSP" --kg run_50000/edges_cui.tsv \
  --icd2cui run_50000/icd2cui.tsv --out run_50000/edges_cui_validated.tsv

python temporal_direction.py --hosp "$MIMIC3_HOSP" \
  --kg run_50000/edges_cui_validated.tsv --icd2cui run_50000/icd2cui.tsv \
  --out run_50000/edges_final.tsv
```

Verify the required artifacts:

```bash
for f in run_50000/llm_triples.jsonl run_50000/edges_sectioned_llm.tsv \
  run_50000/phrase2cui.tsv run_50000/edges_cui.tsv run_50000/icd2cui.tsv \
  run_50000/edges_cui_validated.tsv run_50000/edges_final.tsv
do
  test -s "$f" && echo "OK   $f" || echo "MISS $f"
done
```

## 9. Run the independent literature audit

```bash
python semmeddb_ground_truth.py "$SEMMED" --kg run_50000/edges_final.tsv \
  --out run_50000/semmeddb_causal.tsv \
  --vs-out run_50000/semmeddb_vs_temporal.tsv

python direction_sensitivity.py --src run_50000/semmeddb_vs_temporal.tsv \
  --out run_50000/direction_sensitivity_results.json

python summarize_audit.py --dataset MIMIC-III-50k \
  --lift run_50000/edges_cui_validated.tsv \
  --temporal run_50000/edges_final.tsv \
  --semmed run_50000/semmeddb_vs_temporal.tsv \
  --out run_50000/audit_summary.json
```

## 10. Publication decision gate

The experiment supports an external-validation claim only if the manuscript reports:

1. the same prompt, Qwen checkpoint, temperature, UMLS threshold, lift rule, temporal
   thresholds, and SemMedDB version as MIMIC-IV;
2. temporal commitment coverage, temporal direction accuracy, and LLM direction accuracy
   on the *same committed subset*;
3. SemMedDB coverage and PMID-threshold sensitivity; and
4. all sample sizes and the result direction, including any disagreement with MIMIC-IV.

Do not demand numerical equality. The claim is defensible only if the qualitative result
replicates: temporal coding order remains materially weaker than the extractor/reference
where it commits. If it is competitive or better, report that heterogeneity and soften the
paper's universal language.

## 11. Preserve the reproducibility record

Keep Slurm logs, `audit_summary.json`, the fixed seed, and input/output checksums. Do not
release MIMIC note text or patient-level identifiers; release only permitted derived
artifacts and code/configuration for credentialed users to reproduce the run.
