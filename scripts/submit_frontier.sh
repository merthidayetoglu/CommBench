
module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-cray
module load amd-mixed/5.4.3
module unload darshan-runtime

export MPICH_GPU_SUPPORT_ENABLED=1

salloc -A CMB103 -t 01:00:00 -N 4 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest --switch=1

