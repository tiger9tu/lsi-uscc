#!/bin/bash -l
#SBATCH --account=dongyl0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --job-name=h6_sa_m2_1pct
#SBATCH --output=/home/yuetu/projects/las-luscc/experiments/h6-mlas-m2-frac001-spin-adapted/logs/%x_%A_%a.out
#SBATCH --error=/home/yuetu/projects/las-luscc/experiments/h6-mlas-m2-frac001-spin-adapted/logs/%x_%A_%a.err

set -euo pipefail
cd /home/yuetu/projects/las-luscc
source env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYSCF_MAX_MEMORY=7000
python3 -u experiments/h6-mlas-m2-frac001-spin-adapted/run.py
