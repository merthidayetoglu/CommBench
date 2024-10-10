# create the CommFunctions library
add_library(CommFunctions commbench.cpp commbench.h comm.h util.h)

# Create a variables using option and set default to OFF
option(USE_CUDA "Use CUDA implementation" OFF)
option(USE_XCCL "Use Vendor-provided collective communication library")

# Create a variable for NCCL path
set(XCCL_PATH "" CACHE PATH "Path to the XCCL installation")


if (USE_CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(CommFunctions PRIVATE "PORT_CUDA")
    
    # Include CUDA directories
    target_include_directories(CommFunctions PRIVATE ${CUDA_INCLUDE_DIRS})

    # Check if USE_XCCL is enabled
    if (USE_XCCL)
        # Add NCCL to the linker flags
	target_compile_definitions(CommFunctions PRIVATE "CAP_NCCL")

        # Find NCCL package
        find_library(NCCL_LIB nccl HINTS ${XCCL_PATH}/lib REQUIRED)
        find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${XCCL_PATH}/include REQUIRED)

        if (NCCL_LIB AND NCCL_INCLUDE_DIR)
            # Add NCCL to the linker flags
            target_link_libraries(CommFunctions PRIVATE ${NCCL_LIB})

            # Include NCCL directories
            target_include_directories(CommFunctions PRIVATE ${NCCL_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "NCCL library or include directory not found in the specified path: ${NCCL_PATH}")
        endif()

    endif()
endif()


# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
target_include_directories(CommFunctions
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )

# should we use our own math functions

# library that just does sqrt
#add_library(CommLibrary STATIC
#	      comm.cpp
#	      util.cpp
#              comm.h
#	      comm_impl.h 
#	      util.h
#              )

# link SqrtLibrary to tutorial_compiler_flags
#target_link_libraries(CommLibrary PUBLIC commbench_compiler_flags)

#target_link_libraries(CommFunctions PRIVATE CommLibrary)

# link MathFunctions to tutorial_compiler_flags
#target_link_libraries(CommFunctions PUBLIC CommLibrary tutorial_compiler_flags)
target_link_libraries(CommFunctions PUBLIC commbench_compiler_flags)
