#ifndef DYNAMICGEOMETRY_H
#define DYNAMICGEOMETRY_H
#include "params.hpp"
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
#include <optix_stack_size.h>
#include "common/common.h"

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct dynamicGeometry
{
    dynamicGeometry();
    ~dynamicGeometry();
    OptixDeviceContext optix_context = nullptr;

    size_t temp_buffer_size = 0;
    CUdeviceptr d_temp_buffer = 0;
    CUdeviceptr d_temp_vertices = 0;
    CUdeviceptr d_instances = 0;

    unsigned int triangle_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput ias_instance_input = {};
    OptixBuildInput triangle_input = {};

    OptixTraversableHandle ias_handle;
    OptixTraversableHandle static_gas_handle;
    OptixTraversableHandle deforming_gas_handle;
    OptixTraversableHandle exploding_gas_handle;

    CUdeviceptr d_ias_output_buffer = 0;
    CUdeviceptr d_static_output_buffer;
    CUdeviceptr d_deforming_output_buffer;
    CUdeviceptr d_exploding_output_buffer;

    size_t ias_output_buffer_size = 0;
    size_t static_gas_output_buffer_size = 0;
    size_t deforming_gas_output_buffer_size = 0;
    size_t exploding_gas_output_buffer_size = 0;

    OptixModule module;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;

    OptixProgramGroup raygen_prog_group;
    OptixProgramGroup miss_group = 0;
    OptixProgramGroup hit_group = 0;

    CUstream stream = 0;

    Params params;
    Params *d_params;

    float time = 0.f;
    float last_exploding_sphere_rebuild_time = 0.f;

    OptixShaderBindingTable sbt = {};

    void init();
    void initLaunchParams();
    void initCameraState();
    void createContext();
    void createModule(std::string ptx_filename);
    void createPipeline();
    void createProgramGroups();
    void createSBT();
    sutil::Camera camera;
    bool camera_changed = true;
};

#endif
