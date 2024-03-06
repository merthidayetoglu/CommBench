#!/usr/bin/bash -i
 
rocm_version=5.4.3
 
git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
 
cd aws-ofi-rccl
 
module load libtool
module swap PrgEnv-cray PrgEnv-gnu
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load gcc/12.2.0
module load cray-mpich/8.1.23
export libfabric_path=/opt/cray/libfabric/1.15.2.0/
 
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
 
CC=hipcc CXX=hipcc CFLAGS=-I/opt/rocm-$rocm_version/rccl/include ./configure --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-rccl=/opt/rocm-$rocm_version --enable-trace --prefix=$PWD --with-hip=/opt/rocm-$rocm_version/hip --with-mpi=$MPICH_DIR
 
make
make install
 
echo $PWD
echo "Add the following line in the environment to use the AWS OFI RCCL plugin"
echo "export LD_LIBRARY_PATH="$PWD"/lib:$""LD_LIBRARY_PATH"
