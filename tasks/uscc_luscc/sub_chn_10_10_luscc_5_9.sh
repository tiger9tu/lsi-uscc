#!/bin/bash
#SBATCH --account=dongyl0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.out
#SBATCH --error=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.err

set -euo pipefail

cd /home/yuetu/projects/las-luscc
source /home/yuetu/projects/las-luscc/env.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYSCF_MAX_MEMORY=80000

echo "host=$(hostname) job=${SLURM_JOB_ID} method=luscc cpus=${SLURM_CPUS_PER_TASK}"
python -u tasks/uscc_luscc/chn_10_10_split.py luscc
