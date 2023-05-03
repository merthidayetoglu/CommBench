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
 #include <nccl.h>
// #include <rccl.h>
// #include <sycl.hpp>
// #include <ze_api.h>

// PORTS
 #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL

// CONTROL NCCL CAPABILITY
#if defined(PORT_CUDA) || defined(PORT_HIP)
#define CAP_NCCL
#endif

#include "../comm.h"

void setup_gpu();
void print_args();

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

  if(argc != 8) {print_args(); MPI_Finalize(); return 0;}
  // INPUT PARAMETERS
  int library = atoi(argv[1]);
  int direction = atoi(argv[2]);
  size_t count = atoi(argv[3]);
  int warmup = atoi(argv[4]);
  int numiter = atoi(argv[5]);
  int sender = atoi(argv[6]);
  int recver = atoi(argv[7]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);

    printf("Library: %d\n", library);
    printf("Direction: %d\n", direction);
    printf("Sender: %d\n", sender);
    printf("Recver: %d\n", recver);

    printf("Point-to-point (P2P) count %ld (%ld Bytes)\n", count, count * sizeof(float));
    printf("\n");
  }

  setup_gpu();

  // SETUP NCCL
#ifdef CAP_NCCL
    ncclComm_t comm_nccl;
    ncclUniqueId id;
    if(myid == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm_nccl, numproc, id, myid);
    cudaStream_t sendstream;
    cudaStream_t recvstream;
    for(int p = 0; p < numproc; p++) {
      cudaStreamCreate(&sendstream);
      cudaStreamCreate(&recvstream);
    }
#endif

  float *sendbuf_d;
  float *recvbuf_d;

#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(float));
  cudaMalloc(&recvbuf_d, count * sizeof(float));
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(float));
  hipMalloc(&recvbuf_d, count * sizeof(float));
#elif defined PORT_SYCL
  sycl::queue q(sycl::gpu_selector_v);
  sendbuf_d = sycl::malloc_device<float>(count, q);
  recvbuf_d = sycl::malloc_device<float>(count, q);
#else
  sendbuf_d = new float[count];
  recvbuf_d = new float[count];
#endif

  /*{
    CommBench::Comm<float> bench(MPI_COMM_WORLD, (CommBench::library) library);
    bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);

    bench.report();

    double minTime;
    double medTime;
    double maxTime;
    double avgTime;

    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);

    if(myid == ROOT) {
      double data = count * sizeof(float) / 1.e9;
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data, data / minTime);
      printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data, data / medTime);
      printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data, data / maxTime);
      printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data, data / avgTime);
      printf("\n");
    }
  }
  return 0; */

  MPI_Request sendrequest;
  MPI_Request recvrequest;

  double times[numiter];
  if(myid == ROOT)
    printf("%d warmup iterations (in order):\n", warmup);
  for (int iter = -warmup; iter < numiter; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    switch(library) {
      case 1:
        switch(direction) {
          case 1:
            if(myid == sender) {
              MPI_Isend(sendbuf_d, count, MPI_FLOAT, recver, 0, MPI_COMM_WORLD, &sendrequest);
              MPI_Wait(&sendrequest, MPI_STATUS_IGNORE);
            }
	    if(myid == recver) {
              MPI_Irecv(recvbuf_d, count, MPI_FLOAT, sender, 0, MPI_COMM_WORLD, &recvrequest);
              MPI_Wait(&recvrequest, MPI_STATUS_IGNORE);
            }
            break;
          case 2:
            if(myid == sender) {
              MPI_Isend(sendbuf_d, count, MPI_FLOAT, recver, 0, MPI_COMM_WORLD, &sendrequest);
              MPI_Irecv(recvbuf_d, count, MPI_FLOAT, recver, 0, MPI_COMM_WORLD, &recvrequest);
              MPI_Wait(&sendrequest, MPI_STATUS_IGNORE);
              MPI_Wait(&recvrequest, MPI_STATUS_IGNORE);
            }
            if(myid == recver) {
              MPI_Isend(sendbuf_d, count, MPI_FLOAT, sender, 0, MPI_COMM_WORLD, &sendrequest);
              MPI_Irecv(recvbuf_d, count, MPI_FLOAT, sender, 0, MPI_COMM_WORLD, &recvrequest);
              MPI_Wait(&sendrequest, MPI_STATUS_IGNORE);
              MPI_Wait(&recvrequest, MPI_STATUS_IGNORE);
            }
            break;
        } break;
#ifdef CAP_NCCL
      case 2:
        ncclGroupStart();
        switch(direction) {
          case 1:
            if(myid == sender)
              ncclSend(sendbuf_d, count, ncclFloat32, recver, comm_nccl, sendstream);
            if(myid == recver)
              ncclRecv(recvbuf_d, count, ncclFloat32, sender, comm_nccl, recvstream);
            break;
          case 2:
            if(myid == sender) {
              ncclSend(sendbuf_d, count, ncclFloat32, recver, comm_nccl, sendstream);
              ncclRecv(recvbuf_d, count, ncclFloat32, recver, comm_nccl, recvstream);
            }
            if(myid == recver) {
              ncclSend(sendbuf_d, count, ncclFloat32, sender, comm_nccl, sendstream);
              ncclRecv(recvbuf_d, count, ncclFloat32, sender, comm_nccl, recvstream);
            }
            break;
        }
        ncclGroupEnd();
        cudaStreamSynchronize(sendstream);
        cudaStreamSynchronize(recvstream);
        break;
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    if(iter < 0) {
      if(myid == ROOT)
        printf("warmup: %e\n", time);
    }
    else
      times[iter] = time;
  }

  std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

  if(myid == ROOT) {
    printf("%d measurement iterations (sorted):\n", numiter);
    for(int iter = 0; iter < numiter; iter++) {
      printf("time: %.4e", times[iter]);
      if(iter == 0)
        printf(" -> min\n");
      else if(iter == numiter / 2)
        printf(" -> median\n");
      else if(iter == numiter - 1)
        printf(" -> max\n");
      else
        printf("\n");
    }
    printf("\n");
    double minTime = times[0];
    double medTime = times[numiter / 2];
    double maxTime = times[numiter - 1];
    double avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;
    double data = count * sizeof(float) / 1.e9;
    if(direction == 2)
      data *= 2;
    switch(library) {
      case 1:
        switch(direction) {
          case 1: printf("MPI_Isend / MPI_Irecv (%d -> %d)\n", sender, recver); break;
          case 2: printf("MPI_Isend / MPI_Irecv (%d <-> %d)\n", sender, recver); break;
        } break;
#ifdef CAP_NCCL
      case 2:
        switch(direction) {
          case 1: printf("ncclSend / ncclRecv (%d -> %d)\n", sender, recver); break;
          case 2: printf("ncclSend / ncclRecv (%d <-> %d)\n", sender, recver); break;
        } break;
#endif
    }
    printf("data: %.4e MB\n", data * 1e3);
    printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data, data / minTime);
    printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data, data / medTime);
    printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data, data / maxTime);
    printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data, data / avgTime);
    printf("\n");
  }

#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
#elif defined PORT_SYCL
  sycl::free(sendbuf_d, q);
  sycl::free(recvbuf_d, q);
#else
  delete[] sendbuf_d;
  delete[] recvbuf_d;
#endif

#ifdef CAP_NCCL
  ncclCommDestroy(comm_nccl);
  cudaStreamDestroy(sendstream);
  cudaStreamDestroy(recvstream);
#endif

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
    printf("Point-to-point (pt2pt) test requires seven arguments:\n");
    printf("1. library: 1 for MPI, 2 for NCCL or RCCL\n");
    printf("2. direction: 1 for unidirectional, 2 for bidirectional\n");
    printf("3. count: number of 4-byte elements\n");
    printf("4. warmup: number of warmup rounds\n");
    printf("5. numiter: number of measurement rounds\n");
    printf("6. sender: sender rank index\n");
    printf("7. recver: receiver rank index\n");
    printf("where on can run pt2pt test as\n");
    printf("mpirun ./CommBench library direction count warmup numiter sender recver\n");
    printf("\n");
  }
}
