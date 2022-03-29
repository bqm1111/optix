cmake_minimum_required(VERSION 3.17)
project(simulator LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures")

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
# turn on C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# We use this to be able to link against CUDA::cudart and/or CUDA::cuda_driver
find_package(CUDAToolkit REQUIRED)

# Make sure we have access to Optix7
set(OPTIX_HOME "/home/tainp/Workspace/dat/optix/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX7 REQUIRED)
include_directories(${OptiX7_INCLUDE_DIRS})

include_directories(${OPTIX_HOME}/SDK)
include_directories(${OPTIX_HOME}/SDK/support)
include_directories(${OPTIX_HOME}/SDK/support/GLFW/include)

# ==============================================================================
# Utility function to get PTX compilation & copying working
# ==============================================================================
include(PTXUtilities)

# for common headers
include_directories(./)
include_directories(common)

add_subdirectory(triangle)
add_subdirectory(dynamicGeometry)