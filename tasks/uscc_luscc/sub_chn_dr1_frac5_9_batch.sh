#!/bin/bash
# Submit bisdiazene dr=1.0 LAS-USCC + LAS-LUSCC jobs for FRAC=5-9%.
set -euo pipefail

cd "$(dirname "$0")"

for frac in 0.05 0.06 0.07 0.08 0.09; do
    jobname="chn-dr1.0-frac${frac}"
    sbatch --job-name="${jobname}" \
           --export=ALL,DR=1.0,FRAC=${frac} \
           sub_chn_dr_frac.sh
done
