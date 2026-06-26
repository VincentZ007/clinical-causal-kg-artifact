# Reproduction guide

This guide describes the expected workflow. Paths are examples and should be changed to local paths.

## 1. Environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate ickg-review
```

For GPU extraction, install a vLLM build compatible with your cluster CUDA version if the environment solver does not install it cleanly.

## 2. Data paths

Set local paths:

```bash
export MIMIC4_NOTE=/path/to/mimic-iv-note
export MIMIC4_HOSP=/path/to/mimiciv/3.1/hosp
export MIMIC3_HOSP=/path/to/mimiciii/1.4
export MIMIC3_NOTES=$MIMIC3_HOSP/NOTEEVENTS.csv.gz
export SEMMEDDB_PREDICATION=/path/to/semmedVER43_2021_R_PREDICATION.csv.gz
```

## 3. Main MIMIC-IV pipeline

The main pipeline follows this order:

1. Build the cohort and train/test split.
2. Prepare note sections for extraction.
3. Extract causal triples with the LLM.
4. Parse triples and build surface-form edges.
5. Link entities to UMLS CUIs.
6. Filter candidate edges using co-occurrence lift as patient-level EHR support.
7. Audit direction against temporal order and SemMedDB.
8. Build and evaluate the causal QA benchmark.
9. Regenerate figures and paper tables from aggregate results.

The scripts in `code/mimic_iv/` implement these steps.

The final QA comparison uses `vcrag_causal_prompts.py` and `eval_liftonly.py`.
The key controlled condition is:

```bash
python code/mimic_iv/vcrag_causal_prompts.py --base liftonly --mode demote ...
python code/mimic_iv/eval_liftonly.py ...
```

This evaluates the lift-only retrieval baseline and a conservative causal-demotion variant. The demotion variant preserves the retrieved candidate set and only changes evidence order, so its main interpretation is hallucination reduction at fixed retrieval recall rather than proof of causal truth.

## 4. MIMIC-III external replication

Use the runbook in `code/mimic_iii/MIMIC3_HPC_RUNBOOK_CN.md`.

The expected aggregate output is summarized in `results/aggregate/mimic3_summary.json`.

## 5. Paper build

Upload the contents of `paper/` to Overleaf or compile locally with an LNCS-compatible LaTeX installation.

The figures are included both in `paper/` and `paper/figures/` so the source can compile in simple Overleaf layouts.
