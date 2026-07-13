#!/bin/bash -l
#SBATCH --account=pi-lgagliardi
##SBATCH --qos=lgagliardi-gpu
#SBATCH --qos=lgagliardi
#SBATCH --time=96:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-gpu2
#SBATCH --partition=lgagliardi-ld
#SBATCH --ntasks-per-node=48
#SBATCH --mem=400G
#SBATCH --error=error.%j.err
#SBATCH --job-name=polyene
##SBATCH --gres=gpu:4 #number of GPUs requested. 
module purge
module load rcc
module load slurm

module load python/anaconda-2020.11
module load gcc
module load mkl
module load cmake/4.3
module load hdf5/1.14.3
module load mpich/3.4.3+gcc-10.2.0
module load cuda

export SHARED_APPS=/project/lgagliardi/valayagarawal/Apps
export SHARED_PYSCF=$SHARED_APPS/pyscf
export SHARED_PYSCF_EXT=$SHARED_APPS/pyscf_ext/.pyscf_ext_path
export PERSONAL_MRH=$SHARED_APPS/mrh-gpu
#export PERSONAL_MRH_GPU=$SHARED_APPS/mrh-gpu/mrh/gpu #if you want to run your pyscf/mrh with GPU acceleration
export LD_PRELOAD=${MKLROOT}/lib/intel64/libmkl_core.so:${MKLROOT}/lib/intel64/libmkl_sequential.so:/software/gcc-10.2.0-el8-x86_64/lib64/libgfortran.so.5.0.0

export PYTHONPATH=$SHARED_PYSCF:$PERSONAL_MRH:$PYTHONPATH
#export PYTHONPATH=$SHARED_PYSCF:$PERSONAL_MRH:$PERSONAL_MRH_GPU:$PYTHONPATH  #comment out above line and use this for gpu accelerated runs
export PYSCF_EXT_PATH=$SHARED_PYSCF_EXT:$SHARED_PYSCF
#export CUDA_VISIBLE_DEVICES=0,1,2,3

ulimit -s unlimited
export PYSCF_MAX_MEMORY=400000
/usr/bin/time --verbose python $1
