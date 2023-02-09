
module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-cray
module load rocm

export MPICH_GPU_SUPPORT_ENABLED=1

salloc -A CHM137_crusher -t 00:30:00 -N 2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest
