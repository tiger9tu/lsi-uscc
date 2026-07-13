#!/bin/bash
# Source this file (do not execute) to enter the project environment:
#   source /home/yuetu/projects/las-luscc/env.sh
#
# Loads Lmod modules and activates the project's Python venv (if present).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

module purge
module load python/3.11.5
module load gcc/13.2.0
module load cmake/3.26.3

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYSCF_MAX_MEMORY="${PYSCF_MAX_MEMORY:-8000}"
