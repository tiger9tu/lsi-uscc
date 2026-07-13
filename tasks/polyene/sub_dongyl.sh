#!/bin/bash
#SBATCH --account=dongyl0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.out
#SBATCH --error=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.err

set -euo pipefail

# Activate project env (modules + in-project .venv)
source /home/yuetu/projects/las-luscc/env.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Usage: sbatch -J <name> sub_dongyl.sh <python_script>
if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch -J <name> sub_dongyl.sh <python_script>" >&2
    exit 1
fi
SCRIPT="$1"

cd /home/yuetu/projects/las-luscc

echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} script=${SCRIPT}"
python -u "${SCRIPT}"
