#!/usr/bin/env bash
# Download SemMedDB PREDICATION (needs UMLS UTS api_key) and build the independent
# causal direction gold. Reads the key from pystow config or the UMLS_API_KEY env var,
# so the key never has to appear on the command line.
#
# ONE-TIME key setup (private, key never enters the chat) — put in ~/.config/pystow.ini:
#   [umls]
#   api key: YOUR_UTS_API_KEY
# then:  bash download_semmeddb.sh
#
set -e
export PYSTOW_HOME=$HOME/.cache/pystow
cd .

echo "[1/2] downloading SemMedDB PREDICATION (semmedVER43_2021_R, several GB; 10-30 min)..."
P=$(conda run -n causal-kg python -c "import umls_downloader as u; print(u.download_semmeddb_predication())" | tail -1)
echo "      downloaded: $P"

echo "[2/2] filtering CAUSES/PREDISPOSES/INDUCES to our KG CUIs -> semmeddb_causal.tsv ..."
conda run -n causal-kg python semmeddb_ground_truth.py "$P"
echo "DONE. -> semmeddb_causal.tsv  +  semmeddb_vs_temporal.tsv"
