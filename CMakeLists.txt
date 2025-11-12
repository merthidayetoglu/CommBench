cmake_minimum_required(VERSION 3.24)

# Set the project name and version
project(CommBench VERSION 1.0)

# Find MPI package
find_package(MPI REQUIRED)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# specify the C++ standard
add_library(commbench_compiler_flags INTERFACE)
target_compile_features(commbench_compiler_flags INTERFACE cxx_std_11)

# Add option for GTL and XPMEM
option(USE_GTL "Use GTL library" OFF)
option(USE_XPMEM "Use XPMEM library" OFF)

# Create a variable for GTL path
set(GTL_PATH "" CACHE PATH "Path to the GTL library")

# Check if MPI was found
if (MPI_FOUND)
    message(STATUS "MPI found")

    # Set the MPI compilers
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

    message(STATUS "Using MPI C compiler: ${MPI_C_COMPILER}")
    message(STATUS "Using MPI C++ compiler: ${MPI_CXX_COMPILER}")

    set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>")

    target_compile_options(commbench_compiler_flags INTERFACE
            "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-fopenmp>>"
    )
else()
    # Check if the user enabled GASNET
    if (USE_GASNET)
        message(FATAL_ERROR "GASNET is not supported yet. Please use MPI.")
    else()
        message(FATAL_ERROR "MPI is required but was not found. Please install MPI or enable GASNET with -DUSE_GASNET=ON.")
    endif()
endif()

# Create a variables using option and set default to OFF
option(USE_CUDA "Use CUDA implementation" OFF)
option(USE_HIP "Use CUDA implementation" OFF)
option(USE_XCCL "Use Vendor-provided collective communication library" OFF)
option(USE_BLIT_KERNEL "Use Blit/Copy Kernel for IPC communication" OFF)

option(USE_CALIPER "Enable Caliper benchmarking support" OFF)
option(USE_JSONCPP "Enable support for passing a config file" OFF)
option(PRINT_DEBUG "Enable debug output" OFF)


# Create a variable for NCCL path
set(XCCL_PATH "" CACHE PATH "Path to the XCCL installation")
set(HIP_PATH "" CACHE PATH "Path to the HIP installation")

if (USE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)

    # Option for specifying CUDA architecture
    # set(CUDA_ARCHITECTURE "80" CACHE STRING "Specify the CUDA architecture (e.g., 80 meaning arch=compute_80,code=sm_80)")
    set(CMAKE_CUDA_ARCHITECTURES "80")
    # target_compile_options(commbench_compiler_flags INTERFACE "-ccbin=mpicxx")
elseif (USE_HIP)
    find_package(HIP REQUIRED)
    enable_language(HIP)

    message(STATUS "HIP found: ${HIP_VERSION}")  # Print message if HIP is found
    set(HIP_ARCHITECTURE "gfx942;gfx90A" CACHE STRING "Specify the HIP architecture (e.g., gfx942;gfx90A)")
endif()

if (USE_CALIPER)
    find_package(caliper)
endif()

add_subdirectory(CommFunctions) # add subdirectory after all configuration, may need to be moved down further

# Add your source files and executable here
add_executable(${PROJECT_NAME} main.cpp)
add_executable(bench_p2p bench_p2p.cpp)
add_executable(bench_alltoall bench_alltoall.cpp)
add_executable(bench_broadcast bench_broadcast.cpp)
add_executable(bench_gather bench_gather.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC CommFunctions commbench_compiler_flags)
target_link_libraries(bench_p2p PUBLIC CommFunctions commbench_compiler_flags)
target_link_libraries(bench_alltoall PUBLIC CommFunctions commbench_compiler_flags)
target_link_libraries(bench_broadcast PUBLIC CommFunctions commbench_compiler_flags)
target_link_libraries(bench_gather PUBLIC CommFunctions commbench_compiler_flags)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/CommFunctions DESTINATION include FILES_MATCHING PATTERN "*.h")

if (USE_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bench_p2p.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bench_alltoall.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bench_broadcast.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bench_gather.cpp PROPERTIES LANGUAGE CUDA)
elseif (USE_HIP)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(bench_p2p.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(bench_alltoall.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(bench_broadcast.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(bench_gather.cpp PROPERTIES LANGUAGE HIP)
endif()

if (USE_JSONCPP)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "CONFIG_FILE")
endif()

enable_testing()

add_subdirectory(tests)  
