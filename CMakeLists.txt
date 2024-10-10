cmake_minimum_required(VERSION 3.15)

# Set the project name and version
project(CommBench VERSION 1.0)

# Find MPI package
find_package(MPI REQUIRED)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# specify the C++ standard
add_library(commbench_compiler_flags INTERFACE)
target_compile_features(commbench_compiler_flags INTERFACE cxx_std_11)

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
