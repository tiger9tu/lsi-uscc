#!/bin/sh
#SBATCH --mem=80Gb
#SBATCH --time=100:00:00
#SBATCH --partition=lgagliardi-amd
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=/project/lgagliardi/tuy/qchem/tmp/slurm_%j.out
#SBATCH --error=/project/lgagliardi/tuy/qchem/tmp/slurm_%j.err


export OMP_NUM_THREADS=16
source /project/lgagliardi/tuy/qchem/.venv/bin/activate

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_file> <output_file>" >&2
    exit 1
fi

input="$1"
output="$2"

python "$input" > "$output"
