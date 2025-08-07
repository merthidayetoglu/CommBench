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

void set_device(int device) {
#ifdef PORT_CUDA
  cudaSetDevice(device);
#elif defined PORT_HIP
  hipSetDevice(device);
#endif
  mydevice = device;
}

void setup_gpu() {
  static int init = false;
#ifdef PORT_CUDA
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  #ifdef CAP_NCCL
    if (numproc > deviceCount) {
      fprintf(stderr, "Warning: Using the same device for different ranks of a communicator for NCCL is not supported\n");
    }
  #endif
  int device = myid % deviceCount;
  // cudaSetDevice(device);
  set_device(device);
  if(!init) {
    if(myid == printid)
      fprintf(stderr, "CUDA PORT\n");
    // SET DEVICE
    if(myid == printid)
      fprintf(stderr, "deviceCount: %d\n", deviceCount);
    // REPORT
    if(myid == printid){
      int error = system("nvidia-smi");
      int deviceCount;
      int device;
      cudaGetDevice(&device);
      cudaGetDeviceCount(&deviceCount);
      fprintf(stderr, "Device %d Count: %d\n", device, deviceCount);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp,0);
      fprintf(stderr, "Device %d name: %s\n",0,deviceProp.name);
      fprintf(stderr, "Clock Frequency: %f GHz\n",deviceProp.clockRate/1.e9);
      fprintf(stderr, "Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
      fprintf(stderr, "Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
      fprintf(stderr, "Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
      fprintf(stderr, "Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
      fprintf(stderr, "Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
      fprintf(stderr, "Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
      fprintf(stderr, "Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
      fprintf(stderr, "Warp size: %d\n",deviceProp.warpSize);
      fprintf(stderr, "32-bit Reg. per block: %d\n",deviceProp.regsPerBlock);
      fprintf(stderr, "\n");
    }
  }
#elif defined PORT_HIP
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  // hipSetDevice(device);
  set_device(device);
  if(!init) {
    if(myid == printid)
      fprintf(stderr, "HIP PORT\n");
    //DEVICE MANAGEMENT
    if(myid == printid)
      fprintf(stderr, "deviceCount: %d\n", deviceCount);
    // REPORT
    if(myid == printid) {
      system("rocm-smi");
      int deviceCount;
      int device;
      hipGetDevice(&device);
      hipGetDeviceCount(&deviceCount);
      fprintf(stderr, "Device %d Count: %d\n", device, deviceCount);
      hipDeviceProp_t deviceProp;
      hipGetDeviceProperties(&deviceProp,0);
      fprintf(stderr, "Device %d name: %s\n",0,deviceProp.name);
      fprintf(stderr, "Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
      fprintf(stderr, "Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
      fprintf(stderr, "32-bit Reg. per block: %d\n",deviceProp.regsPerBlock);
      fprintf(stderr, "Warp size: %d\n",deviceProp.warpSize);
      fprintf(stderr, "Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
      fprintf(stderr, "Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
      fprintf(stderr, "Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
      fprintf(stderr, "Clock frequency: %d khz\n",deviceProp.clockRate);
      fprintf(stderr, "Global memory frequency: %d khz\n", deviceProp.memoryClockRate);
      fprintf(stderr, "Global memory bus width: %d bits\n", deviceProp.memoryBusWidth);
      fprintf(stderr, "Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
      fprintf(stderr, "Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
      fprintf(stderr, "Number of multi-processors: %d\n", deviceProp.multiProcessorCount);
      fprintf(stderr, "L2 cache size: %d\n", deviceProp.l2CacheSize);
      fprintf(stderr, "Max. threads per multi-processor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
      fprintf(stderr, "Compute mode: %d\n", deviceProp.computeMode);
      fprintf(stderr, "Device-side clock instruction rate: %d khz\n", deviceProp.clockInstructionRate);
      fprintf(stderr, "\n");
    }
  }
#elif defined PORT_SYCL
  if(!init) {
    if(CommBench::myid == CommBench::printid)
      fprintf(stderr, "SYCL PORT\n");
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
      // for(int d = 0; d < deviceCount; ++d) {
      for(int d = 0; d < 1; ++d) {
        ze_device_properties_t device_properties;
        zeDeviceGetProperties(allDevices[d], &device_properties);
        if(CommBench::myid == CommBench::printid)
        {
          if(ZE_DEVICE_TYPE_GPU == device_properties.type)
            fprintf(stderr, "driverCount %d deviceCount %d GPU\n", driverCount, deviceCount);
          else
            fprintf(stderr, "GPU not found!\n");
          fprintf(stderr, "type %d\n", device_properties.type);
          fprintf(stderr, "vendorId %d\n", device_properties.vendorId);
          fprintf(stderr, "deviceId %d\n", device_properties.deviceId);
          fprintf(stderr, "flags %d\n", device_properties.flags);
          fprintf(stderr, "subdeviceId %d\n", device_properties.subdeviceId);
          fprintf(stderr, "coreClockRate %d\n", device_properties.coreClockRate);
          fprintf(stderr, "maxMemAllocSize %ld\n", device_properties.maxMemAllocSize);
          fprintf(stderr, "maxHardwareContexts %d\n", device_properties.maxHardwareContexts);
          fprintf(stderr, "maxCommandQueuePriority %d\n", device_properties.maxCommandQueuePriority);
          fprintf(stderr, "numThreadsPerEU %d\n", device_properties.numThreadsPerEU);
          fprintf(stderr, "physicalEUSimdWidth %d\n", device_properties.physicalEUSimdWidth);
          fprintf(stderr, "numSubslicesPerSlice %d\n", device_properties.numEUsPerSubslice);
          fprintf(stderr, "numSlices %d\n", device_properties.numSlices);
          fprintf(stderr, "timerResolution %ld\n", device_properties.timerResolution);
          fprintf(stderr, "timestampValidBits %d\n", device_properties.timestampValidBits);
          fprintf(stderr, "kernelTimestampValidBits %d\n", device_properties.kernelTimestampValidBits);
          //for(int j = 0; j < ZE_MAX_DEVICE_UUID_SIZE; j++)
          //  fprintf(stderr, "uuid %d\n", device_properties.uuid.id[j]);
          fprintf(stderr, "name %s\n", device_properties.name);
          fprintf(stderr, "\n");
        }
      }
      /*ze_bool_t test = false;
      zeDeviceCanAccessPeer(allDevices[0], allDevices[1], &test);
      fprintf(stderr, "can access peer %d\n", test);*/
      delete[] allDevices;
    }
    delete[] allDrivers;
  }
#else
  if(!init)
    if(CommBench::myid == CommBench::printid)
      fprintf(stderr, "CPU VERSION\n");
#endif
  init = true;
}
