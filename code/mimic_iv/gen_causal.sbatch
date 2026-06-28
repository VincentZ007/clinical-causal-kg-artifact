#!/bin/bash
#SBATCH --job-name=vcrag_gen_causal
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:a100_1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=vcrag_gen_causal-%j.log
set -e
source /sw/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vllm_qwen}"
cd "${ICKG_BASE:-$PWD}"
echo "host: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES  start: $(date)"
python gen_local.py prompts_causal.jsonl answers_causal.jsonl 8
echo "done: $(date)"
