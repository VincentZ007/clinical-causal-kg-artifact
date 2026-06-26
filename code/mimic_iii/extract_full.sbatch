#!/bin/bash
#SBATCH --job-name=ickg-mimic3
# If your cluster requires a reservation or account, add those SBATCH lines here.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00              # ~7h expected + margin
#SBATCH --output=ickg-mimic3-%j.out

set -euo pipefail
source /sw/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate vllm_qwen
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export VLLM_LOGGING_LEVEL=WARNING

nvidia-smi -L
python extract_llm_full.py "${INPUT:?Set INPUT before sbatch}" "${OUTPUT:?Set OUTPUT before sbatch}"
