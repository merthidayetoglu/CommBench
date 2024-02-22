export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

module use /soft/modulefiles
module load spack-pe-gcc
module load cmake

module  use /soft/modulefiles
module load spack-pe-gcc thapi

qsub -l select=1 -l walltime=01:00:00 -A CSC249ADCD01_CNDA -q EarlyAppAccess -I

