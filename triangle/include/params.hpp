#ifndef PARAMS_H
#define PARAMS_H
#include <cuda_runtime.h>
#include <optix.h>
#include <vector>
#include <stdint.h>

struct RayGenData
{
    
};

struct HitGroupData
{
    float color;
    float3 center;
    float radius;
};

struct MissData
{
    float3 bg_color;
};

struct Params
{
    uchar4* image;
    unsigned int image_width;
    unsigned int image_height;
    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

#endif
