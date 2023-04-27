
module load gcc
module load cuda
module load job-step-viewer

bsub -q debug -alloc_flags gpudefault -W 01:00 -nnodes 2 -P CHM137 -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash

