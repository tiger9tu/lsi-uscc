#!/bin/bash
# Submit 3x4 = 12 (dr, frac) jobs for bisdiazene LAS-USCC + LAS-LUSCC.
set -euo pipefail

cd "$(dirname "$0")"

for dr in 0.0 1.0 2.0; do
    for frac in 0.01 0.02 0.03 0.04; do
        # job name -> chn-dr0.0-frac0.01 etc. (slurm uses %x for this).
        jobname="chn-dr${dr}-frac${frac}"
        sbatch --job-name="${jobname}" \
               --export=ALL,DR=${dr},FRAC=${frac} \
               sub_chn_dr_frac.sh
    done
done
