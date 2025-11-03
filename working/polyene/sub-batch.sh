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

source /home/tuy/repo/.venv/bin/activate

# 一个小函数，减少重复
run() {
  local py="$1" base="$2"
  srun -N1 -n1 -c ${SLURM_CPUS_PER_TASK} --exclusive \
       /usr/bin/time -v -o ${base}_time.txt \
       python "${py}" > ${base}_result.txt 2> ${base}_stderr.txt &
}

run c4_631g.py c4_631g
run c6_631g.py c6_631g
run c8_631g.py c8_631g
run c10_631g.py c10_631g

wait
echo "All tasks finished."
