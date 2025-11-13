#!/bin/bash
#SBATCH --job-name=stil-batch
#SBATCH --partition=lgagliardi-amd
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --nodes=1
#SBATCH --ntasks=1               # 我们用多次 srun 启动 step，不用一次性 ntasks>1
#SBATCH --cpus-per-task=4        # 每个脚本占 4 核
#SBATCH --mem=80G
#SBATCH --time=100:00:00

# 根据 cpus-per-task 设定 OpenMP
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

source /project/lgagliardi/tuy/qchem/.venv/bin/activate

# 一个小函数，减少重复
run() {
  local py="$1" base="$2"
  srun -N1 -n1 -c ${SLURM_CPUS_PER_TASK} --exclusive \
       /usr/bin/time -v -o ${base}_time.txt \
       python "${py}" > ${base}_result.txt 2> ${base}_stderr.txt &
}

run stil-triplet-001.py  stil_triplet_001
run stil-triplet-60.py   stil_triplet_60
run stil-triplet-90.py   stil_triplet_90
run stil-triplet-120.py  stil_triplet_120
run stil-triplet-180.py  stil_triplet_180

wait
echo "All tasks finished."
