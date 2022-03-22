#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include <optix.h>
#include "common/common.h"

void optixLogCallback(unsigned int level, const char *tag, const char *message,
                      void *cbdata) {
  std::cout << "Optix Log[" << level << "][" << tag << "]: '" << message
            << "'\n";
}

void printActiveCudaDevices(void) {
  // Query number of available CUDA devices
  int num_cuda_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_cuda_devices));
  
  // Print available CUDA devices' names
  std::cout << "Active CUDA Devices: \n";
  for (int i = 0; i < num_cuda_devices; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    std::cout << "\tDevice " << i << ": " << prop.name << "\n";
  }
  std::cout << "\n";
}

std::string getFileContent(std::string const &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file for reading: '" + path + "'");

  std::stringstream stream;
  stream << file.rdbuf();

  if (file.bad() || file.fail())
    throw std::runtime_error("Error reading file content from: '" + path + "'");

  return stream.str();
}

void cuda_free_event_callback(cudaStream_t stream, cudaError_t status,
                              void *userData) {
  cudaFree(userData);
}

