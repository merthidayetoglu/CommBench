/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h> // for printf
#include <stdlib.h> // for atoi
#include <cstring> // for memcpy
#include <algorithm> // for sort
#include <mpi.h>
#include <omp.h>

#define ROOT 0

// HEADERS
// #include <nccl.h>
// #include <rccl.h>
// #include <sycl.hpp>
// #include <ze_api.h>

// PORTS
// #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL

#include "comm.h"

void setup_gpu();
void print_args();

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

int main(int argc, char *argv[])
{
  // INITIALIZE MPI+OPENMP
  int myid;
  int numproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  int numthread;
  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();
  //char machine_name[MPI_MAX_PROCESSOR_NAME];
  //int name_len = 0;
  //MPI_Get_processor_name(machine_name, &name_len);
  //printf("myid %d %s\n",myid, machine_name);

  if(argc != 10) {print_args(); MPI_Finalize(); return 0;}
  // INPUT PARAMETERS
  int library = atoi(argv[1]);
  int pattern = atoi(argv[2]);
  int direction = atoi(argv[3]);
  size_t count = atoi(argv[4]);
  int warmup = atoi(argv[5]);
  int numiter = atoi(argv[6]);
  int numgpu = atoi(argv[7]);
  int groupsize = atoi(argv[8]);
  int subgroupsize = atoi(argv[9]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);
    printf("Number of GPUs %d\n", numgpu);
    printf("Group Size: %d\n", groupsize);
    printf("Subgroup Size: %d\n", subgroupsize);

    printf("Library: %d\n", library);
    printf("Pattern: %d\n", pattern);
    printf("Direction: %d\n", direction);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Peer-to-peer count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("send buffer: %lu (%.2f GB) recv buffer: %lu (%.2f GB)\n", count, count * sizeof(Type) / 1.e9, count * numproc, count * numproc * sizeof(Type) / 1.e9);
    printf("\n");
  }

  setup_gpu();

  if(pattern == 0)
#include "test_P2P.h"
  if(pattern == 1)
#include "test_RAIL.h"
  if(pattern == 2)
#include "test_DENSE.h"
  if(pattern == 3)
#include "test_FAN.h"

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

void setup_gpu() {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

#ifdef PORT_CUDA
  if(myid == ROOT)
    printf("CUDA PORT\n");
  // SET DEVICE
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  cudaSetDevice(device);
  // REPORT
  if(myid == ROOT){
    system("nvidia-smi");
    int deviceCount;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceCount(&deviceCount);
    printf("Device %d Count: %d\n", device, deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    printf("Device %d name: %s\n",0,deviceProp.name);
    printf("Clock Frequency: %f GHz\n",deviceProp.clockRate/1.e9);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("32-bit Reg. per block: %d\n",deviceProp.regsPerBlock);
    printf("\n");
  }
#elif defined PORT_HIP
  if(myid == ROOT)
    printf("HIP PORT\n");
  //DEVICE MANAGEMENT
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  if(myid == ROOT)
    printf("deviceCount: %d\n", deviceCount);
  hipSetDevice(device);
  // REPORT
  if(myid == ROOT) {
    system("rocm-smi");
    int deviceCount;
    int device;
    hipGetDevice(&device);
    hipGetDeviceCount(&deviceCount);
    printf("Device %d Count: %d\n", device, deviceCount);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp,0);
    printf("Device %d name: %s\n",0,deviceProp.name);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("32-bit Reg. per block: %d\n",deviceProp.regsPerBlock); 
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Clock frequency: %d khz\n",deviceProp.clockRate);
    printf("Global memory frequency: %d khz\n", deviceProp.memoryClockRate);
    printf("Global memory bus width: %d bits\n", deviceProp.memoryBusWidth);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Number of multi-processors: %d\n", deviceProp.multiProcessorCount);
    printf("L2 cache size: %d\n", deviceProp.l2CacheSize);
    printf("Max. threads per multi-processor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Compute mode: %d\n", deviceProp.computeMode);
    printf("Device-side clock instruction rate: %d khz\n", deviceProp.clockInstructionRate);
    printf("\n");
  }
#elif defined PORT_SYCL
  if(myid == ROOT)
    printf("SYCL PORT\n");
  // Initialize the driver
  zeInit(0);
  // Discover all the driver instances
  uint32_t driverCount = 0;
  zeDriverGet(&driverCount, nullptr);
  ze_driver_handle_t* allDrivers = new ze_driver_handle_t[driverCount];
  zeDriverGet(&driverCount, allDrivers);
  // Find a driver instance with a GPU device
  ze_driver_handle_t hDriver = nullptr;
  ze_device_handle_t hDevice = nullptr;
  for(int i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    zeDeviceGet(allDrivers[i], &deviceCount, nullptr);
    ze_device_handle_t* allDevices = new ze_device_handle_t[deviceCount];
    zeDeviceGet(allDrivers[i], &deviceCount, allDevices);
    for(int d = 0; d < deviceCount; ++d) {
      ze_device_properties_t device_properties;
      zeDeviceGetProperties(allDevices[d], &device_properties);
      if(myid == ROOT)
      {
        if(ZE_DEVICE_TYPE_GPU == device_properties.type)
          printf("driverCount %d deviceCount %d GPU\n", driverCount, deviceCount);
        else
          printf("GPU not found!\n");
        printf("type %d\n", device_properties.type);
        printf("vendorId %d\n", device_properties.vendorId);
        printf("deviceId %d\n", device_properties.deviceId);
        printf("flags %d\n", device_properties.flags);
        printf("subdeviceId %d\n", device_properties.subdeviceId);
        printf("coreClockRate %d\n", device_properties.coreClockRate);
        printf("maxMemAllocSize %ld\n", device_properties.maxMemAllocSize);
        printf("maxHardwareContexts %d\n", device_properties.maxHardwareContexts);
        printf("maxCommandQueuePriority %d\n", device_properties.maxCommandQueuePriority);
        printf("numThreadsPerEU %d\n", device_properties.numThreadsPerEU);
        printf("physicalEUSimdWidth %d\n", device_properties.physicalEUSimdWidth);
        printf("numSubslicesPerSlice %d\n", device_properties.numEUsPerSubslice);
        printf("numSlices %d\n", device_properties.numSlices);
        printf("timerResolution %ld\n", device_properties.timerResolution);
        printf("timestampValidBits %d\n", device_properties.timestampValidBits);
        printf("kernelTimestampValidBits %d\n", device_properties.kernelTimestampValidBits);
        //for(int j = 0; j < ZE_MAX_DEVICE_UUID_SIZE; j++)
	//  printf("uuid %d\n", device_properties.uuid.id[j]);
	printf("name %s\n", device_properties.name);
        printf("\n");
      }
    }
    delete[] allDevices;
  }
  delete[] allDrivers;
#else
  if(myid == ROOT)
    printf("CPU VERSION\n");
#endif
}

void print_args() {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  if(myid == ROOT) {
    printf("\n");
    printf("CommBench requires nine arguments:\n");
    printf("1. library: 0 for IPC, 1 for MPI, 2 for NCCL or RCCL\n");
    printf("2. pattern: 1 for Rail, 2 for Dense, 3 for Fan\n");
    printf("3. direction: 1 for unidirectional, 2 for bidirectional, 3 for omnidirectional\n");
    printf("4. count: number of 4-byte elements\n");
    printf("5. warmup: number of warmup rounds\n");
    printf("6. numiter: number of measurement rounds\n");
    printf("7. p: number of processors\n");
    printf("8. g: group size\n");
    printf("9. k: subgroup size\n");
    printf("where on can run CommBench as\n");
    printf("mpirun ./CommBench library pattern direction count warmup numiter p g k\n");
    printf("\n");
  }
}
