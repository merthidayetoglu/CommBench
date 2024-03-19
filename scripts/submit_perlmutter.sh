
module reset
module load nccl
module load nvhpc

salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4301
