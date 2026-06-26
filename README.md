# Anonymous clinical causal KG artifact

This repository contains the code, paper source, figures, and aggregate results for the submitted manuscript.

The repository intentionally does not include MIMIC, PhysioNet, UMLS, SemMedDB, patient-level prompts, note text, patient identifiers, or derived patient-level benchmark files. Those materials inherit access restrictions from the original datasets.

The artifact is intended for double-anonymous review. Please use the anonymized mirror link rather than the source GitHub URL when citing this repository during review.

## What is included

- `code/mimic_iv/`: scripts used for the main MIMIC-IV causal KG audit and QA evaluation.
- `code/mimic_iii/`: scripts and HPC runbook for the MIMIC-III external replication.
- `results/aggregate/`: non-patient-level summary results used in the paper.
- `paper/`: LNCS paper source, bibliography, style files, and figures.
- `data/`: notes on restricted data access and expected local paths.

## Quick start

1. Read `DATA_ACCESS.md`.
2. Install the environment from `environment.yml`.
3. Follow `REPRODUCE.md` for the MIMIC-IV and MIMIC-III pipelines.
4. Compile `paper/main.tex` with the included LNCS files.

The scripts regenerate the restricted intermediate files locally after the user obtains the required dataset credentials.

## Current aggregate QA result

The main downstream result is the lift-only retrieval baseline plus conservative causal demotion. The demotion step keeps the same top-12 retrieved candidates, but moves candidates that fail the causal screen to the end of the evidence list. It is an evidence-ordering step, not a claim that the remaining candidates are proven causal.

On the held-out causal-QA subset, lift-only retrieval reaches 35.7% overall accuracy and 86.3% retrieval recall. Adding causal demotion reaches 37.2% overall accuracy, keeps recall at 86.3%, and reduces unsupported hallucinated answers from 17.3% to 6.1%. The overall accuracy gain is a positive but non-significant trend; the robust effect is the hallucination reduction.
