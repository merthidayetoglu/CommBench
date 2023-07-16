
module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-cray
module load amd-mixed
module unload darshan-runtime

export MPICH_GPU_SUPPORT_ENABLED=1

salloc -A CHM137 -t 01:00:00 -N 4 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest --switch=1

