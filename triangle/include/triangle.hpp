#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <common/common.h>
#include <params.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <memory.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <common/common.h>
#include <optix_stack_size.h>
#include "utils.h"
#include "opencv2/opencv.hpp"
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct Triangle
{
    Triangle();
    ~Triangle();
    OptixDeviceContext optix_context = nullptr;
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OptixProgramGroup ocean_hitgroup_prog_group = nullptr;

    OptixPipeline pipeline = nullptr;

    OptixShaderBindingTable sbt = {};

    Params params;
    Params *d_params;
    sutil::Camera camera;

    cudaStream_t stream;
    // Functions to work with Optix
    void init();
    void configCamera();
    void createContext();
    void buildGAS();
    void createModule(const std::string ptx_filename);
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
    void launch();

    uint32_t m_width;
    uint32_t m_height;
};

#endif