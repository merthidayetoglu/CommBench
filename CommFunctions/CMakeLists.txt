# create the CommFunctions library
add_library(CommFunctions commbench.cpp commbench.h comm.h util.h)

# Create a variables using option and set default to OFF
option(USE_CUDA "Use CUDA implementation" OFF)
option(USE_HIP "Use CUDA implementation" OFF)
option(USE_XCCL "Use Vendor-provided collective communication library")

# Create a variable for NCCL path
set(XCCL_PATH "" CACHE PATH "Path to the XCCL installation")
set(HIP_PATH "" CACHE PATH "Path to the HIP installation")

# Option for specifying HIP architecture
set(HIP_ARCHITECTURE "gfx942;gfx90A" CACHE STRING "Specify the HIP architecture (e.g., gfx942;gfx90A)")

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
            message(FATAL_ERROR "NCCL library or include directory not found in the specified path: ${XCCL_PATH}")
        endif()

    endif()
endif()


# Find HIP package
if (USE_HIP)

    find_package(HIP REQUIRED)

    message(STATUS "HIP found: ${HIP_VERSION}")  # Print message if HIP is found
    target_compile_definitions(CommFunctions PRIVATE "PORT_HIP")

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


target_include_directories(CommFunctions
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )

target_link_libraries(CommFunctions PUBLIC commbench_compiler_flags)
