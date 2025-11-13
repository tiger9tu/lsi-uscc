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


# /usr/bin/time -v  python c6_631g_gradportion.py 0.5 > c6_631g_result_gradportion05.txt
# /usr/bin/time -v  python c6_631g_gradportion.py 1 > c6_631g_result_gradportion1.txt
# /usr/bin/time -v  python c6_631g_gradportion.py 2 > c6_631g_result_gradportion2.txt
# /usr/bin/time -v  python c6_631g_gradportion.py 4 > c6_631g_result_gradportion4.txt
# /usr/bin/time -v  python c6_631g_gradportion.py -0.5 > c6_631g_result_gradportionneg05.txt
# /usr/bin/time -v  python c6_631g_gradportion.py -1 > c6_631g_result_gradportionneg1.txt
# /usr/bin/time -v  python c6_631g_gradportion.py -2 > c6_631g_result_gradportionneg2.txt
# /usr/bin/time -v  python c6_631g_gradportion.py -4 > c6_631g_result_gradportionneg4.txt
/usr/bin/time -v  python c6_631g.py 2 >  c6_631g_result_t2.txt
# /usr/bin/time -v  python c6_631g_rand.py 0.5 >  c6_631g_result_rand05.txt
# /usr/bin/time -v  python c6_631g_rand.py 1 >    c6_631g_result_rand1.txt
# /usr/bin/time -v  python c6_631g_rand.py 2 >    c6_631g_result_rand2.txt
# /usr/bin/time -v  python c6_631g_rand.py 4 >    c6_631g_result_rand4.txt
# /usr/bin/time -v  python c6_631g_rand.py 10 >   c6_631g_result_rand10.txt

