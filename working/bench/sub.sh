#!/bin/sh
#SBATCH --mem=80Gb
#SBATCH --time=100:00:00
#SBATCH --partition=lgagliardi-amd
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24


export OMP_NUM_THREADS=16
source /project/lgagliardi/tuy/qchem/.venv/bin/activate


python new.py > new_res.txt
# /usr/bin/time -v  python c4_631g_rand.py 0.5 >  c4_631g_result_rand05.txt
# /usr/bin/time -v  python c4_631g_rand.py 1 >    c4_631g_result_rand1.txt
# /usr/bin/time -v  python c4_631g_rand.py 2 >    c4_631g_result_rand2.txt
# /usr/bin/time -v  python c4_631g_rand.py 4 >    c4_631g_result_rand4.txt
# /usr/bin/time -v  python c4_631g_rand.py 10 >   c4_631g_result_rand10.txt


