#!/bin/sh
#SBATCH --mem=80Gb
#SBATCH --time=100:00:00
#SBATCH --partition=lgagliardi-amd
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

export OMP_NUM_THREADS=16
source /home/tuy/repo/.venv/bin/activate
python stilbene-001_energy_comparison.py > stil001out.txt 

