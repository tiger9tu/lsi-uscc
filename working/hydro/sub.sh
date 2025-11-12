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


# /usr/bin/time -v  python h4_gradportion_amps.py 0.5 > h4_result_gradportion05.txt
# /usr/bin/time -v  python h4_gradportion_amps.py 1   > h4_result_gradportion1.txt
# /usr/bin/time -v  python h4_gradportion_amps.py 2   > h4_result_gradportion2.txt
# /usr/bin/time -v  python h4_gradportion_amps.py 4   > h4_result_gradportion4.txt
# /usr/bin/time -v  python h4_gradportion_amps.py -0.5    > h4_result_gradportionneg05.txt
# /usr/bin/time -v  python h4_gradportion_amps.py -1      > h4_result_gradportionneg1.txt
# /usr/bin/time -v  python h4_gradportion_amps.py -2.0    > h4_result_gradportionneg2.txt
# /usr/bin/time -v  python h4_gradportion_amps.py -4.0    > h4_result_gradportionneg4.txt
/usr/bin/time -v  python h4_randamp.py 0.1 > h4_result_h4_randamp01.txt
/usr/bin/time -v  python h4_randamp.py 0.5 > h4_result_h4_randamp05.txt
/usr/bin/time -v  python h4_randamp.py 1   > h4_result_h4_randamp1.txt
/usr/bin/time -v  python h4_randamp.py 2   > h4_result_h4_randamp2.txt
/usr/bin/time -v  python h4_randamp.py 4   > h4_result_h4_randamp4.txt


