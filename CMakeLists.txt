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





# UNIT TEST SECTION
# Create the tests/bin directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/unit_tests)

# Set the output directory for the test executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_tests)

# Add the MPI/IPC_PUT/IPC_GET/IPC_BLIT/XCCL test executable
add_executable(mpi_p2p_test tests/mpi_p2p_test.cpp)
target_link_libraries(mpi_p2p_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(mpi_gather_test tests/mpi_gather_test.cpp)
target_link_libraries(mpi_gather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(mpi_scatter_test tests/mpi_scatter_test.cpp)
target_link_libraries(mpi_scatter_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(mpi_bcast_test tests/mpi_bcast_test.cpp)
target_link_libraries(mpi_bcast_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(mpi_alltoall_test tests/mpi_alltoall_test.cpp)
target_link_libraries(mpi_alltoall_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(mpi_allgather_test tests/mpi_allgather_test.cpp)
target_link_libraries(mpi_allgather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)

if (USE_CUDA OR USE_HIP)
add_executable(ipc_put_p2p_test tests/ipc_put_p2p_test.cpp)
target_link_libraries(ipc_put_p2p_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_put_gather_test tests/ipc_put_gather_test.cpp)
target_link_libraries(ipc_put_gather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_put_scatter_test tests/ipc_put_scatter_test.cpp)
target_link_libraries(ipc_put_scatter_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_put_bcast_test tests/ipc_put_bcast_test.cpp)
target_link_libraries(ipc_put_bcast_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_put_alltoall_test tests/ipc_put_alltoall_test.cpp)
target_link_libraries(ipc_put_alltoall_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_put_allgather_test tests/ipc_put_allgather_test.cpp)
target_link_libraries(ipc_put_allgather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)


add_executable(ipc_get_p2p_test tests/ipc_get_p2p_test.cpp)
target_link_libraries(ipc_get_p2p_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_get_gather_test tests/ipc_get_gather_test.cpp)
target_link_libraries(ipc_get_gather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_get_scatter_test tests/ipc_get_scatter_test.cpp)
target_link_libraries(ipc_get_scatter_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_get_bcast_test tests/ipc_get_bcast_test.cpp)
target_link_libraries(ipc_get_bcast_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_get_alltoall_test tests/ipc_get_alltoall_test.cpp)
target_link_libraries(ipc_get_alltoall_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
add_executable(ipc_get_allgather_test tests/ipc_get_allgather_test.cpp)
target_link_libraries(ipc_get_allgather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)



if (USE_XCCL)
   add_executable(xccl_p2p_test tests/xccl_p2p_test.cpp)
   target_link_libraries(xccl_p2p_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
   add_executable(xccl_gather_test tests/xccl_gather_test.cpp)
   target_link_libraries(xccl_gather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
   add_executable(xccl_scatter_test tests/xccl_scatter_test.cpp)
   target_link_libraries(xccl_scatter_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
   add_executable(xccl_bcast_test tests/xccl_bcast_test.cpp)
   target_link_libraries(xccl_bcast_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
   add_executable(xccl_alltoall_test tests/xccl_alltoall_test.cpp)
   target_link_libraries(xccl_alltoall_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)
   add_executable(xccl_allgather_test tests/xccl_allgather_test.cpp)
   target_link_libraries(xccl_allgather_test PUBLIC CommFunctions commbench_compiler_flags OpenMP::OpenMP_CXX)

endif()
endif()


# Check if USE_GTL is enabled for the ALL test and add the linking flag
if (USE_GTL)
    target_link_libraries(tests/bin/mpi_p2p_test PRIVATE mpi_gtl_hsa)
    target_link_libraries(mpi_gather_test PRIVATE mpi_gtl_hsa)
    target_link_libraries(mpi_scatter_test PRIVATE mpi_gtl_hsa)
    target_link_libraries(mpi_bcast_test PRIVATE mpi_gtl_hsa)
    target_link_libraries(mpi_alltoall_test PRIVATE mpi_gtl_hsa)
    target_link_libraries(mpi_allgather_test PRIVATE mpi_gtl_hsa)
    if (USE_CUDA OR USE_HIP)
       target_link_libraries(ipc_put_p2p_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_put_gather_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_put_scatter_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_put_bcast_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_put_alltoall_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_put_allgather_test PRIVATE mpi_gtl_hsa)

       target_link_libraries(ipc_get_p2p_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_get_gather_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_get_scatter_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_get_bcast_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_get_alltoall_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(ipc_get_allgather_test PRIVATE mpi_gtl_hsa)
       
       if (USE_XCCL)
       target_link_libraries(xccl_p2p_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(xccl_gather_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(xccl_scatter_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(xccl_bcast_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(xccl_alltoall_test PRIVATE mpi_gtl_hsa)
       target_link_libraries(xccl_allgather_test PRIVATE mpi_gtl_hsa)
       endif()
    endif()    
    message(STATUS "Linking Tests with GTL library: -lmpi_gtl_hsa")

    # Check if GTL_PATH is specified
    if (GTL_PATH)
        target_link_directories(tests/bin/mpi_p2p_test PRIVATE ${GTL_PATH}/lib)
        target_link_directories(mpi_gather_test PRIVATE ${GTL_PATH}/lib)
        target_link_directories(mpi_scatter_test PRIVATE ${GTL_PATH}/lib)
        target_link_directories(mpi_bcast_test PRIVATE ${GTL_PATH}/lib)
        target_link_directories(mpi_alltoall_test PRIVATE ${GTL_PATH}/lib)
        target_link_directories(mpi_allgather_test PRIVATE ${GTL_PATH}/lib)
	if (USE_CUDA OR USE_HIP)
        	target_link_directories(ipc_put_p2p_test PRIVATE ${GTL_PATH}/lib)
		target_link_directories(ipc_put_gather_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_put_scatter_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_put_bcast_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_put_alltoall_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_put_allgather_test PRIVATE ${GTL_PATH}/lib)

		target_link_directories(ipc_get_p2p_test PRIVATE ${GTL_PATH}/lib)	
		target_link_directories(ipc_get_gather_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_get_scatter_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_get_bcast_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_get_alltoall_test PRIVATE ${GTL_PATH}/lib)
                target_link_directories(ipc_get_allgather_test PRIVATE ${GTL_PATH}/lib)
		if (USE_XCCL)
		   target_link_directories(xccl_p2p_test PRIVATE ${GTL_PATH}/lib)
		   target_link_directories(xccl_gather_test PRIVATE ${GTL_PATH}/lib)
                   target_link_directories(xccl_scatter_test PRIVATE ${GTL_PATH}/lib)
                   target_link_directories(xccl_bcast_test PRIVATE ${GTL_PATH}/lib)
                   target_link_directories(xccl_alltoall_test PRIVATE ${GTL_PATH}/lib)
                   target_link_directories(xccl_allgather_test PRIVATE ${GTL_PATH}/lib)
       		endif()
	endif()
	message(STATUS "Adding GTL path to Test linker: ${GTL_PATH}/lib")
    else()
        message(WARNING "GTL_PATH is not specified for Test. Default library path will be used.")
    endif()
endif()

# Check if USE_XPMEM is enabled for the ALL test and add the linking flag
if (USE_XPMEM)
    target_link_libraries(tests/bin/mpi_p2p_test PRIVATE xpmem)
    target_link_libraries(mpi_gather_test PRIVATE xpmem)
    target_link_libraries(mpi_scatter_test PRIVATE xpmem)
    target_link_libraries(mpi_bcast_test PRIVATE xpmem)
    target_link_libraries(mpi_alltoall_test PRIVATE xpmem)
    target_link_libraries(mpi_allgather_test PRIVATE xpmem)
    if (USE_CUDA OR USE_HIP)
       target_link_libraries(ipc_put_p2p_test PRIVATE xpmem)
       target_link_libraries(ipc_put_gather_test PRIVATE xpmem)
       target_link_libraries(ipc_put_scatter_test PRIVATE xpmem)
       target_link_libraries(ipc_put_bcast_test PRIVATE xpmem)
       target_link_libraries(ipc_put_alltoall_test PRIVATE xpmem)
       target_link_libraries(ipc_put_allgather_test PRIVATE xpmem)
       
       target_link_libraries(ipc_get_p2p_test PRIVATE xpmem)  
       target_link_libraries(ipc_get_gather_test PRIVATE xpmem)
       target_link_libraries(ipc_get_scatter_test PRIVATE xpmem)
       target_link_libraries(ipc_get_bcast_test PRIVATE xpmem)
       target_link_libraries(ipc_get_alltoall_test PRIVATE xpmem)
       target_link_libraries(ipc_get_allgather_test PRIVATE xpmem)
       if (USE_XCCL)
           target_link_libraries(xccl_p2p_test PRIVATE xpmem)
           target_link_libraries(xccl_gather_test PRIVATE xpmem)
           target_link_libraries(xccl_scatter_test PRIVATE xpmem)
           target_link_libraries(xccl_bcast_test PRIVATE xpmem)
           target_link_libraries(xccl_alltoall_test PRIVATE xpmem)
           target_link_libraries(xccl_allgather_test PRIVATE xpmem)
       	endif()
    endif()
    message(STATUS "Linking MPI test with XPMEM library: -lxpmem")
endif()

# Enable testing
enable_testing()

# Add a test for the MPI program
add_test(NAME MPI_P2P_Test
         COMMAND ${MPIEXEC} -n 2 $<TARGET_FILE:mpi_p2p_test>
         )
set_tests_properties(MPI_P2P_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME MPI_GATHER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:mpi_gather_test>
         )
set_tests_properties(MPI_GATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME MPI_SCATTER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:mpi_scatter_test>
         )
set_tests_properties(MPI_SCATTER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME MPI_BCAST_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:mpi_bcast_test>
         )
set_tests_properties(MPI_BCAST_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME MPI_ALLTOALL_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:mpi_alltoall_test>
         )
set_tests_properties(MPI_ALLTOALL_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

add_test(NAME MPI_ALLGATHER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:mpi_allgather_test>
         )
set_tests_properties(MPI_ALLGATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")


if (USE_CUDA OR USE_HIP)
add_test(NAME IPC_PUT_P2P_Test
         COMMAND ${MPIEXEC} -n 2 $<TARGET_FILE:ipc_put_p2p_test>
         )
 set_tests_properties(IPC_PUT_P2P_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_PUT_GATHER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_put_gather_test>
         )
 set_tests_properties(IPC_PUT_GATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_PUT_SCATTER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_put_scatter_test>
         )
 set_tests_properties(IPC_PUT_SCATTER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_PUT_BCAST_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_put_bcast_test>
         )
 set_tests_properties(IPC_PUT_BCAST_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_PUT_ALLTOALL_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_put_alltoall_test>
         )
 set_tests_properties(IPC_PUT_ALLTOALL_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_PUT_ALLGATHER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_put_allgather_test>
         )
 set_tests_properties(IPC_PUT_ALLGATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")



 add_test(NAME IPC_GET_P2P_Test
	COMMAND ${MPIEXEC} -n 2 $<TARGET_FILE:ipc_get_p2p_test>
         )
 set_tests_properties(IPC_GET_P2P_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_GET_GATHER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_get_gather_test>
         )
 set_tests_properties(IPC_GET_GATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_GET_SCATTER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_get_scatter_test>
         )
 set_tests_properties(IPC_GET_SCATTER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_GET_BCAST_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_get_bcast_test>
         )
 set_tests_properties(IPC_GET_BCAST_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_GET_ALLTOALL_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_get_alltoall_test>
         )
 set_tests_properties(IPC_GET_ALLTOALL_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 add_test(NAME IPC_GET_ALLGATHER_Test
         COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:ipc_get_allgather_test>
         )
 set_tests_properties(IPC_GET_ALLGATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")



 if (USE_XCCL)
	 add_test(NAME XCCL_P2P_Test
		COMMAND ${MPIEXEC} -n 2 $<TARGET_FILE:xccl_p2p_test>
         )
 	set_tests_properties(XCCL_P2P_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 	add_test(NAME XCCL_GATHER_Test
         	COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:xccl_gather_test>
         )
	 set_tests_properties(XCCL_GATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

	 add_test(NAME XCCL_SCATTER_Test
	 COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:xccl_scatter_test>
         )
 	set_tests_properties(XCCL_SCATTER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 	add_test(NAME XCCL_BCAST_Test
         	COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:xccl_bcast_test>
         )
 	set_tests_properties(XCCL_BCAST_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 	add_test(NAME XCCL_ALLTOALL_Test
         	COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:xccl_alltoall_test>
         )
 	set_tests_properties(XCCL_ALLTOALL_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")

 	add_test(NAME XCCL_ALLGATHER_Test
         	COMMAND ${MPIEXEC} -n 3 $<TARGET_FILE:xccl_allgather_test>
         )
 	set_tests_properties(XCCL_ALLGATHER_Test PROPERTIES PASS_REGULAR_EXPRESSION "PASSED!")
 endif()
 
endif()
