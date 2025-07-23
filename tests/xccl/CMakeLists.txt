add_library(xccl_tests INTERFACE)
target_link_libraries(xccl_tests INTERFACE tests)

if (USE_CUDA)
    set_source_files_properties(xccl_allgather_test.cpp xccl_alltoall_test.cpp xccl_bcast_test.cpp xccl_gather_test.cpp xccl_p2p_test.cpp xccl_scatter_test.cpp xccl_test.cpp PROPERTIES LANGUAGE CUDA)
elseif (USE_HIP)
    set_source_files_properties(xccl_allgather_test.cpp xccl_alltoall_test.cpp xccl_bcast_test.cpp xccl_gather_test.cpp xccl_p2p_test.cpp xccl_scatter_test.cpp xccl_test.cpp PROPERTIES LANGUAGE HIP)
endif()

add_executable(xccl_p2p_test xccl_p2p_test.cpp)
target_link_libraries(xccl_p2p_test PRIVATE xccl_tests)
add_executable(xccl_gather_test xccl_gather_test.cpp)
target_link_libraries(xccl_gather_test PRIVATE xccl_tests)
add_executable(xccl_scatter_test xccl_scatter_test.cpp)
target_link_libraries(xccl_scatter_test PRIVATE xccl_tests)
add_executable(xccl_bcast_test xccl_bcast_test.cpp)
target_link_libraries(xccl_bcast_test PRIVATE xccl_tests)
add_executable(xccl_alltoall_test xccl_alltoall_test.cpp)
target_link_libraries(xccl_alltoall_test PRIVATE xccl_tests)
add_executable(xccl_allgather_test xccl_allgather_test.cpp)
target_link_libraries(xccl_allgather_test PRIVATE xccl_tests)

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
