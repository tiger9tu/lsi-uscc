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

if [ -z "${DR:-}" ] || [ -z "${FRAC:-}" ]; then
    echo "DR and FRAC must be set in the environment" >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} dr=${DR} frac=${FRAC} cpus=${SLURM_CPUS_PER_TASK}"
python -u tasks/uscc_luscc/chn_dr_frac.py --dr "${DR}" --frac "${FRAC}"
