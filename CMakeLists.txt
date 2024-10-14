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

    add_subdirectory(CommFunctions)


else()
    # Check if the user enabled GASNET
    if (USE_GASNET)
        message(FATAL_ERROR "GASNET is not supported yet. Please use MPI.")
    else()
        message(FATAL_ERROR "MPI is required but was not found. Please install MPI or enable GASNET with -DUSE_GASNET=ON.")
    endif()
endif()

# Add your source files and executable here
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)

# Check if USE_GTL is enabled and add the linking flag
if (USE_GTL)
    target_link_libraries(${PROJECT_NAME} PRIVATE mpi_gtl_hsa)
    message(STATUS "Linking with GTL library: -lmpi_gtl_hsa")

    # Check if GTL_PATH is specified
    if (GTL_PATH)
        target_link_directories(${PROJECT_NAME} PRIVATE ${GTL_PATH}/lib)
        message(STATUS "Adding GTL path to linker: ${GTL_PATH}/lib")
    else()
        message(WARNING "GTL_PATH is not specified. Default library path will be used.")
    endif()
endif()

# Check if USE_XPMEM is enabled and add the linking flag
if (USE_XPMEM)
    target_link_libraries(${PROJECT_NAME} PRIVATE xpmem)
    message(STATUS "Linking with XPMEM library: -lxpmem")
endif()
