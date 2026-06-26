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
