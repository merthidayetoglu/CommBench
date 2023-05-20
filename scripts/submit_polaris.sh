
#qsub -A GRACE -q single-gpu -n 1 -t 60 -I
#qsub -q debug -A GRACE -n 2 -t 60 -I --attrs="filesystems=home,theta-fs0"
qsub -A GRACE -I -l select=2 -l filesystems=home -l walltime=1:00:00 -q debug
#qsub -A hp-ptycho -n 1 -t 120 -I
#qsub -A TomoDev --queue debug-cache-quad -n 1 -t 60 -I
#qsub -A FFTBench --attrs mcdram=cache:numa=snc4 -n 128 -t 30 --mode script run.sh

#module load nccl
#module load nvhpc
