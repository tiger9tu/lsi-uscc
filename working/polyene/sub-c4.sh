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


# /usr/bin/time -v  python c4_631g_gradportion.py 0.5 > c4_631g_result_gradportion05.txt
# /usr/bin/time -v  python c4_631g_gradportion.py 1 > c4_631g_result_gradportion1.txt
# /usr/bin/time -v  python c4_631g_gradportion.py 2 > c4_631g_result_gradportion2.txt
# /usr/bin/time -v  python c4_631g_gradportion.py 4 > c4_631g_result_gradportion4.txt
# /usr/bin/time -v  python c4_631g_gradportion.py -0.5 > c4_631g_result_gradportionneg05.txt
# /usr/bin/time -v  python c4_631g_gradportion.py -1 > c4_631g_result_gradportionneg1.txt
# /usr/bin/time -v  python c4_631g_gradportion.py -2 > c4_631g_result_gradportionneg2.txt
# /usr/bin/time -v  python c4_631g_gradportion.py -4 > c4_631g_result_gradportionneg4.txt

# /usr/bin/time -v  python c4_631g_gradsign.py 0.01 >  c4_631g_result_gradsign001.txt
# /usr/bin/time -v  python c4_631g_gradsign.py 0.05 >    c4_631g_result_gradsign005
# /usr/bin/time -v  python c4_631g_gradsign.py 0.1 >    c4_631g_result_gradsign01
# /usr/bin/time -v  python c4_631g_gradsign.py 0.25 >    c4_631g_result_gradsign025
# /usr/bin/time -v  python c4_631g_gradsign.py 0.5 > c4_631g_result_gradsign05
# /usr/bin/time -v  python c4_631g_gradsign.py 1 >   c4_631g_result_gradsign1
# /usr/bin/time -v  python c4_631g_gradsign.py 2 >   c4_631g_result_gradsign2

# /usr/bin/time -v  python c4_631g_gradsign.py -0.01 >  c4_631g_result_gradsignneg001.txt
# /usr/bin/time -v  python c4_631g_gradsign.py -0.05 >    c4_631g_result_gradsignneg005
# /usr/bin/time -v  python c4_631g_gradsign.py -0.1 >    c4_631g_result_gradsignneg01
# /usr/bin/time -v  python c4_631g_gradsign.py -0.25 >    c4_631g_result_gradsignneg025
# /usr/bin/time -v  python c4_631g_gradsign.py -0.5 > c4_631g_result_gradsignneg05
# /usr/bin/time -v  python c4_631g_gradsign.py -1 >   c4_631g_result_gradsignneg1
# /usr/bin/time -v  python c4_631g_gradsign.py -2 >   c4_631g_result_gradsignneg2

/usr/bin/time -v  python c4_631g.py 0.05 >  c4_631g_result_t005.txt
# /usr/bin/time -v  python c4_631g_rand.py 0.5 >  c4_631g_result_rand05.txt
# /usr/bin/time -v  python c4_631g_rand.py 1 >    c4_631g_result_rand1.txt
# /usr/bin/time -v  python c4_631g_rand.py 2 >    c4_631g_result_rand2.txt
# /usr/bin/time -v  python c4_631g_rand.py 4 >    c4_631g_result_rand4.txt
# /usr/bin/time -v  python c4_631g_rand.py 10 >   c4_631g_result_rand10.txt


