#!/usr/bin/env bash
# Download only the MIMIC-IV v3.1 hosp tables needed for the causal-KG project.
# Usage:  bash download_hosp.sh <your_physionet_username>
# Prompts once for your PhysioNet password (not stored, not echoed).
set -euo pipefail

USER_ID="${1:?usage: bash download_hosp.sh <physionet_username>}"
DEST="/path/to/MIMIC"
BASE="https://physionet.org/files/mimiciv/3.1/hosp"
FILES=(
  admissions.csv.gz
  patients.csv.gz
  diagnoses_icd.csv.gz
  d_icd_diagnoses.csv.gz
  prescriptions.csv.gz
)

read -rsp "PhysioNet password for ${USER_ID}: " PW; echo
cd "$DEST"
for f in "${FILES[@]}"; do
  echo ">>> $f"
  # -x preserves the physionet.org/files/mimiciv/3.1/hosp/ path; -c resumes
  wget -c -x --user "$USER_ID" --password "$PW" "$BASE/$f"
done
echo
echo "DONE -> $DEST/physionet.org/files/mimiciv/3.1/hosp/"
ls -lh "$DEST/physionet.org/files/mimiciv/3.1/hosp/" 2>/dev/null || true
