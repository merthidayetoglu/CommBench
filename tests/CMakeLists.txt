file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/unit_tests)

# Set the output directory for the test executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_tests)

add_library(tests INTERFACE)
target_link_libraries(tests INTERFACE CommFunctions OpenMP::OpenMP_CXX)

if (USE_GTL)
    target_link_libraries(tests INTERFACE mpi_gtl_hsa)

    if (GTL_PATH)
        target_link_directories(tests INTERFACE ${GTL_PATH}/lib)
        message(STATUS "Adding GTL path to Test linker: ${GTL_PATH}/lib")
    else()
        message(WARNING "GTL_PATH is not specified for Test. Default library path will be used.")
    endif()
endif()

if (USE_XPMEM)
    target_link_libraries(tests INTERFACE xpmem)
    message(STATUS "Linking MPI test with XPMEM library: -lxpmem")
endif()

add_subdirectory(mpi)

if (USE_CUDA OR USE_HIP)
    add_subdirectory(ipc)
endif()

if (USE_XCCL)
    add_subdirectory(xccl)
endif()