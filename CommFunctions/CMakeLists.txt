# create the CommFunctions library
# add_library(CommFunctions commbench.cpp commbench.h comm.h util.h validate.h)
add_library(CommFunctions INTERFACE commbench.h)
target_link_libraries(CommFunctions INTERFACE MPI::MPI_CXX)

#[=[
# Create a variables using option and set default to OFF
option(USE_CUDA "Use CUDA implementation" OFF)
option(USE_HIP "Use CUDA implementation" OFF)
option(USE_XCCL "Use Vendor-provided collective communication library" OFF)
option(USE_BLIT_KERNEL "Use Blit/Copy Kernel for IPC communication" OFF)


# Create a variable for NCCL path
set(XCCL_PATH "" CACHE PATH "Path to the XCCL installation")
set(HIP_PATH "" CACHE PATH "Path to the HIP installation")

# Option for specifying HIP architecture
set(HIP_ARCHITECTURE "gfx942;gfx90A" CACHE STRING "Specify the HIP architecture (e.g., gfx942;gfx90A)")

# Option for specifying CUDA architecture
set(CUDA_ARCHITECTURE "80" CACHE STRING "Specify the CUDA architecture (e.g., 80 meaning arch=compute_80,code=sm_80)")

if (USE_CUDA)
    # find_package(CUDA REQUIRED)
    # enable_language(CUDA)
    set_source_files_properties(commbench.cpp commbench.h PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(CommFunctions PRIVATE "PORT_CUDA")

    if(USE_BLIT_KERNEL)
	    target_compile_definitions(CommFunctions PRIVATE "IPC_kernel")
    endif()

    # Include CUDA directories
    target_include_directories(CommFunctions PRIVATE ${CUDA_INCLUDE_DIRS})
    
    # Set CUDA architecture flags
    if (NOT CUDA_ARCHITECTURE STREQUAL "")
        set_property(TARGET CommFunctions PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE})
    
    else()
	if (DEFINED CMAKE_CUDA_PLATFORM)
		message(STATUS "CMAKE_CUDA_PLATFORM: ${CMAKE_CUDA_PLATFORM}")
		set_property(TARGET CommFunctions PROPERTY CUDA_ARCHITECTURES CMAKE_CUDA_PLATFORM)
        else()
		message(WARNING "CMAKE_CUDA_PLATFORM is not defined. Defaulting to compute_80,code=sm_80")
		set_property(TARGET CommFunctions PROPERTY CUDA_ARCHITECTURES "80")
        endif()


    endif()
    
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
            message(FATAL_ERROR "NCCL library or include directory not found in the specified path: ${XCCL_PATH}")
        endif()

    endif()
endif()


# Find HIP package
if (USE_HIP)

    find_package(HIP REQUIRED)

    message(STATUS "HIP found: ${HIP_VERSION}")  # Print message if HIP is found
    target_compile_definitions(CommFunctions PRIVATE "PORT_HIP")
    
    if(USE_BLIT_KERNEL)
	    target_compile_definitions(CommFunctions PRIVATE "IPC_kernel")
    endif()

    # Include HIP directories
    target_include_directories(CommFunctions PRIVATE ${HIP_INCLUDE_DIRS})

    target_compile_definitions(CommFunctions PRIVATE "__HIP_PLATFORM_AMD__")

    # Set the HIP architecture based on the user option, introspection or default
    if (NOT HIP_ARCHITECTURE STREQUAL "")
            set_property(TARGET CommFunctions PROPERTY HIP_ARCHITECTURES ${HIP_ARCHITECTURE})
    else()
        if (DEFINED CMAKE_HIP_PLATFORM)
                message(STATUS "CMAKE_HIP_PLATFORM: ${CMAKE_HIP_PLATFORM}")
                set_property(TARGET CommFunctions PROPERTY HIP_ARCHITECTURES CMAKE_HIP_PLATFORM)
        else()
                message(WARNING "CMAKE_HIP_PLATFORM is not defined. Defaulting to gfx942;gfx90A")
                set_property(TARGET CommFunctions PROPERTY HIP_ARCHITECTURES "gfx942;gfxX90A")
        endif()
    endif()

    # Check if USE_RCCL is enabled
    if (USE_XCCL)

        target_compile_definitions(CommFunctions PRIVATE "CAP_NCCL")

        # Find RCCL package
        find_library(RCCL_LIB rccl HINTS ${XCCL_PATH}/lib REQUIRED)

        if (RCCL_LIB)
            # Add RCCL to the linker flags
            target_link_libraries(CommFunctions PRIVATE ${RCCL_LIB})
        else()
            message(FATAL_ERROR "RCCL library or include directory not found in the specified path: ${XCCL_PATH}")
        endif()
    endif()
endif()
]=]

if (USE_CUDA)
    set_source_files_properties(commbench.h PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(CommFunctions INTERFACE "PORT_CUDA")

    # Include CUDA directories
    target_include_directories(CommFunctions INTERFACE ${CUDA_INCLUDE_DIRS})
    
    # Set CUDA architecture flags
    if (NOT CUDA_ARCHITECTURE STREQUAL "")
        set_property(TARGET CommFunctions PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE})
    else()
	if (DEFINED CMAKE_CUDA_PLATFORM)
		message(STATUS "CMAKE_CUDA_PLATFORM: ${CMAKE_CUDA_PLATFORM}")
		set_property(TARGET CommFunctions PROPERTY CUDA_ARCHITECTURES CMAKE_CUDA_PLATFORM)
        else()
		message(WARNING "CMAKE_CUDA_PLATFORM is not defined. Defaulting to compute_80,code=sm_80")
		set_property(TARGET CommFunctions PROPERTY CUDA_ARCHITECTURES "80")
        endif()
    endif()
    
    # Check if USE_XCCL is enabled
    if (USE_XCCL)
        # Find NCCL package
        find_library(NCCL_LIB nccl HINTS ${XCCL_PATH}/lib REQUIRED)
        find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${XCCL_PATH}/include REQUIRED)

        if (NCCL_LIB AND NCCL_INCLUDE_DIR)
            # Add NCCL to the linker flags
            target_link_libraries(CommFunctions INTERFACE ${NCCL_LIB})

            # Include NCCL directories
            target_include_directories(CommFunctions INTERFACE ${NCCL_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "NCCL library or include directory not found in the specified path: ${XCCL_PATH}")
        endif()

    endif()
elseif (USE_HIP)
    set_source_files_properties(commbench.h PROPERTIES LANGUAGE HIP)
    target_compile_definitions(CommFunctions INTERFACE "PORT_HIP")

    # Include HIP directories
    target_include_directories(CommFunctions INTERFACE ${HIP_INCLUDE_DIRS})

    target_compile_definitions(CommFunctions INTERFACE "__HIP_PLATFORM_AMD__")

    # Set the HIP architecture based on the user option, introspection or default
    if (NOT HIP_ARCHITECTURE STREQUAL "")
            set_property(TARGET CommFunctions PROPERTY HIP_ARCHITECTURES ${HIP_ARCHITECTURE})
    else()
        if (DEFINED CMAKE_HIP_PLATFORM)
                message(STATUS "CMAKE_HIP_PLATFORM: ${CMAKE_HIP_PLATFORM}")
                set_property(TARGET CommFunctions PROPERTY HIP_ARCHITECTURES CMAKE_HIP_PLATFORM)
        else()
                message(WARNING "CMAKE_HIP_PLATFORM is not defined. Defaulting to gfx942;gfx90A")
                set_property(TARGET CommFunctions PROPERTY HIP_ARCHITECTURES "gfx942;gfxX90A")
        endif()
    endif()

    # Check if USE_RCCL is enabled
    if (USE_XCCL)
        # Find RCCL package
        find_library(RCCL_LIB rccl HINTS ${XCCL_PATH}/lib REQUIRED)

        if (RCCL_LIB)
            # Add RCCL to the linker flags
            target_link_libraries(CommFunctions INTERFACE ${RCCL_LIB})
        else()
            message(FATAL_ERROR "RCCL library or include directory not found in the specified path: ${XCCL_PATH}")
        endif()
    endif()
endif()

if (USE_CUDA OR USE_HIP)
    if(USE_BLIT_KERNEL)
	    target_compile_definitions(CommFunctions INTERFACE "IPC_kernel")
    endif()

    if (USE_XCCL)
        target_compile_definitions(CommFunctions INTERFACE "CAP_NCCL")
    endif()
endif()

target_include_directories(CommFunctions
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )

target_link_libraries(CommFunctions INTERFACE commbench_compiler_flags)
