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


# /usr/bin/time -v  python stil-001.py > stil_001_result.txt
# /usr/bin/time -v  python stil-60.py > stil_60_result.txt
# /usr/bin/time -v  python stil-90.py > stil_90_result.txt
# /usr/bin/time -v  python stil-120.py > stil_120_result.txt
/usr/bin/time -v  python stil-180.py > stil_180_result.txt


