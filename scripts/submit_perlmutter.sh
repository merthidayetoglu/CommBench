
module reset
module load nvhpc
module load nccl/2.18.3-cu12

salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4301
