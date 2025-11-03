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


# /usr/bin/time -v  python c4_631g.py > c4_631g_result.txt
# /usr/bin/time -v  python c6_631g.py > c6_631g_result.txt
/usr/bin/time -v  python c8_631g.py > c8_631g_result.txt
# /usr/bin/time -v  python c10_631g.py > c10_631g_result.txt


