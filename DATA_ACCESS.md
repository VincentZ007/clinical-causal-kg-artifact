# Data access and redistribution policy

This artifact is designed for reproducibility without redistributing restricted clinical data.

## Required external resources

- MIMIC-IV Note and MIMIC-IV hosp/core tables from PhysioNet.
- MIMIC-III Clinical Database from PhysioNet for the external replication.
- UMLS access for entity linking.
- SemMedDB PREDICATION file for the literature-derived validation signal.

## Not included in this repository

The following files are intentionally excluded:

- raw clinical notes and MIMIC tables;
- patient identifiers, note identifiers, admission identifiers, or subject-level rows;
- patient-grounded QA benchmark JSONL files;
- prompts and model outputs generated from patient records;
- UMLS or SemMedDB raw releases;
- full edge tables that may preserve patient-derived concepts or identifiers.

Reviewers with credentialed access can regenerate these files using the provided scripts.
