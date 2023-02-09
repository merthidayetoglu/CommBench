
module reset
module load nccl

salloc --account=bbkf-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=2 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=16 \
  --exclusive --mem=0 --time=01:00:00
