
module reset
module load craype-accel-amd-gfx908
module load PrgEnv-cray
module load rocm

## These must be set before running
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=CMA

salloc -A CHM137 -p caar -t 00:30:00 -N 2 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=closest
