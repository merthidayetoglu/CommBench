add_library(mpi_tests INTERFACE)
target_link_libraries(mpi_tests INTERFACE tests)

if (USE_CUDA)
    set_source_files_properties(mpi_p2p_test.cpp mpi_gather_test.cpp mpi_scatter_test.cpp mpi_bcast_test.cpp mpi_alltoall_test.cpp mpi_allgather_test.cpp PROPERTIES LANGUAGE CUDA)
elseif (USE_HIP)
    set_source_files_properties(mpi_p2p_test.cpp mpi_gather_test.cpp mpi_scatter_test.cpp mpi_bcast_test.cpp mpi_alltoall_test.cpp mpi_allgather_test.cpp DIRECTORY . PROPERTIES LANGUAGE HIP)
endif()

add_executable(mpi_p2p_test mpi_p2p_test.cpp)
target_link_libraries(mpi_p2p_test PRIVATE mpi_tests)
add_executable(mpi_gather_test mpi_gather_test.cpp)
target_link_libraries(mpi_gather_test PRIVATE mpi_tests)
add_executable(mpi_scatter_test mpi_scatter_test.cpp)
target_link_libraries(mpi_scatter_test PRIVATE mpi_tests)
add_executable(mpi_bcast_test mpi_bcast_test.cpp)
target_link_libraries(mpi_bcast_test PRIVATE mpi_tests)
add_executable(mpi_alltoall_test mpi_alltoall_test.cpp)
target_link_libraries(mpi_alltoall_test PRIVATE mpi_tests)
add_executable(mpi_allgather_test mpi_allgather_test.cpp)
target_link_libraries(mpi_allgather_test PRIVATE mpi_tests)

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