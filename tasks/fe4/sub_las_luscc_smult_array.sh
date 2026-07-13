#!/bin/bash -l
#SBATCH --account=dongyl0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=100:00:00
#SBATCH --job-name=fe4_luscc_s
#SBATCH --array=0-6%2
#SBATCH --output=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%A_%a.out
#SBATCH --error=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%A_%a.err
#SBATCH --mail-user=tuyue3@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /home/yuetu/projects/las-luscc
mkdir -p tasks/fe4/data
source env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYSCF_MAX_MEMORY=120000

SMULTS=(3 5 7 9 11 13 19)
NROOTS=(15 20 22 21 17 12 15)
SMULT=${SMULTS[$SLURM_ARRAY_TASK_ID]}
NROOT=${NROOTS[$SLURM_ARRAY_TASK_ID]}

python -u tasks/fe4/run_las_luscc.py \
  --frac 0.01 \
  --smult "${SMULT}" \
  --nroots "${NROOT}" \
  --max-memory 120000
