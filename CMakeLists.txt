cmake_minimum_required(VERSION 3.18)
project(GPU_Matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture (optional fallback if not passed via -DCUTLASS_NVCC_ARCHS)
if(NOT DEFINED CUTLASS_NVCC_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Add CUTLASS directory
include_directories(${CMAKE_SOURCE_DIR}/cutlass/include)

# Enable all warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add your source files here
add_executable(gpu_matmul src/main.cu)

# Optional: enable CUDA-specific flags
set_target_properties(gpu_matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
