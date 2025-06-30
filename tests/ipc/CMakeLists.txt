add_library(ipc_tests INTERFACE)
target_link_libraries(ipc_tests INTERFACE tests)

if (USE_CUDA)
    set_source_files_properties(ipc_get_allgather_test.cpp ipc_get_alltoall_test.cpp ipc_get_bcast_test.cpp ipc_get_gather_test.cpp ipc_get_p2p_test.cpp ipc_get_scatter_test.cpp ipc_get_test.cpp ipc_put_allgather_test.cpp ipc_put_alltoall_test.cpp ipc_put_bcast_test.cpp ipc_put_gather_test.cpp ipc_put_p2p_test.cpp ipc_put_scatter_test.cpp ipc_put_test.cpp PROPERTIES LANGUAGE CUDA)
elseif (USE_HIP)
    set_source_files_properties(ipc_get_allgather_test.cpp ipc_get_alltoall_test.cpp ipc_get_bcast_test.cpp ipc_get_gather_test.cpp ipc_get_p2p_test.cpp ipc_get_scatter_test.cpp ipc_get_test.cpp ipc_put_allgather_test.cpp ipc_put_alltoall_test.cpp ipc_put_bcast_test.cpp ipc_put_gather_test.cpp ipc_put_p2p_test.cpp ipc_put_scatter_test.cpp ipc_put_test.cpp PROPERTIES LANGUAGE HIP)
endif()

add_executable(ipc_put_p2p_test ipc_put_p2p_test.cpp)
target_link_libraries(ipc_put_p2p_test PRIVATE ipc_tests)
add_executable(ipc_put_gather_test ipc_put_gather_test.cpp)
target_link_libraries(ipc_put_gather_test PRIVATE ipc_tests)
add_executable(ipc_put_scatter_test ipc_put_scatter_test.cpp)
target_link_libraries(ipc_put_scatter_test PRIVATE ipc_tests)
add_executable(ipc_put_bcast_test ipc_put_bcast_test.cpp)
target_link_libraries(ipc_put_bcast_test PRIVATE ipc_tests)
add_executable(ipc_put_alltoall_test ipc_put_alltoall_test.cpp)
target_link_libraries(ipc_put_alltoall_test PRIVATE ipc_tests)
add_executable(ipc_put_allgather_test ipc_put_allgather_test.cpp)
target_link_libraries(ipc_put_allgather_test PRIVATE ipc_tests)


add_executable(ipc_get_p2p_test ipc_get_p2p_test.cpp)
target_link_libraries(ipc_get_p2p_test PRIVATE ipc_tests)
add_executable(ipc_get_gather_test ipc_get_gather_test.cpp)
target_link_libraries(ipc_get_gather_test PRIVATE ipc_tests)
add_executable(ipc_get_scatter_test ipc_get_scatter_test.cpp)
target_link_libraries(ipc_get_scatter_test PRIVATE ipc_tests)
add_executable(ipc_get_bcast_test ipc_get_bcast_test.cpp)
target_link_libraries(ipc_get_bcast_test PRIVATE ipc_tests)
add_executable(ipc_get_alltoall_test ipc_get_alltoall_test.cpp)
target_link_libraries(ipc_get_alltoall_test PRIVATE ipc_tests)
add_executable(ipc_get_allgather_test ipc_get_allgather_test.cpp)
target_link_libraries(ipc_get_allgather_test PRIVATE ipc_tests)

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