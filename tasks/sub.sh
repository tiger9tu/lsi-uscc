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

set -e  # err out on first error
export OMP_NUM_THREADS=16
source /project/lgagliardi/tuy/qchem/.venv/bin/activate

# -------------------------
# parsing arguments
# -------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)
            input="$2"
            shift 2
            ;;
        --output)
            output="$2"
            shift 2
            ;;
        --arg1)
            arg1="$2"
            shift 2
            ;;
        --arg2)
            arg2="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# -------------------------
# argument checking
# -------------------------
if [ -z "$input" ] || [ -z "$output" ] || [ -z "$arg1" ] || [ -z "$arg2" ]; then
    echo "Usage:"
    echo "  sbatch sub.sh --input <input.py> --output <out.txt> --arg1 <val1> --arg2 <val2>"
    exit 1
fi

# -------------------------
# execute the python script
# -------------------------
python "$input"  "$arg1" "$arg2" > "$output"
