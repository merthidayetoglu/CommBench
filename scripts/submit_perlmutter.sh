
module reset
module load nvhpc

salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4301
