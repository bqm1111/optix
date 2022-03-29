#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <cuda_runtime.h>
#include <optix.h>
#include <vector>
#include <stdint.h>

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};

struct HitGroupData
{
    float3 color;
};

struct MissData
{
    float4 bg_color;
};

struct Params
{
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    float3 eye, U, V, W;
    OptixTraversableHandle handle;
    int subframe_index;
};

#endif