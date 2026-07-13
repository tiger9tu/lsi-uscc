#!/bin/bash
#SBATCH --account=dongyl0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=fe3_luscc5
#SBATCH --mail-user=tuyue3@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.out
#SBATCH --error=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.err

set -euo pipefail

cd /home/yuetu/projects/las-luscc
source /home/yuetu/projects/las-luscc/env.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYSCF_MAX_MEMORY=120000

echo "host=$(hostname) job=${SLURM_JOB_ID} script=tasks/fe3/las_luscc_from_saved.py cpus=${SLURM_CPUS_PER_TASK}"
python -u tasks/fe3/las_luscc_from_saved.py --frac 0.05 --max-memory 120000
