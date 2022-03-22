#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <optix.h>

#define OPTIX_CHECK(error)                                                     \
  {                                                                            \
    if (error != OPTIX_SUCCESS)                                                \
      std::cerr << __FILE__ << ":" << __LINE__ << " Optix Error: '"            \
                << optixGetErrorString(error) << "'\n";                        \
  }

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << cudaGetErrorString(error) << "'\n";                         \
  }

#define CUDA_DRIVER_CHECK(error)                                               \
  {                                                                            \
    if (error != CUDA_SUCCESS) {                                               \
      const char *error_str = nullptr;                                         \
      cuGetErrorString(error, &error_str);                                     \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << error_str << "'\n";                                         \
    }                                                                          \
  }

// This is used to construct the path to the PTX files
#ifdef _WIN32
#ifdef NDEBUG
#define BUILD_DIR "Release"
#else
#define BUILD_DIR "Debug"
#endif
#else
#define BUILD_DIR "./"
#endif
#define OBJ_DIR "../resources/"

template <typename T> struct SbtRecord {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

// template specialization for empty records
template <> struct SbtRecord<void> {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void optixLogCallback(unsigned int level, const char *tag, const char *message,
                      void *cbdata);

void printActiveCudaDevices(void);

std::string getFileContent(std::string const &path);

void cuda_free_event_callback(cudaStream_t stream, cudaError_t status,
                              void *userData);

#endif