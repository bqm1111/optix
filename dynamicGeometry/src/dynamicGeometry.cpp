#include "dynamicGeometry.h"

// Scene data

const int32_t g_tesselation_resolution = 128;
const float g_exploding_rebuild_frequency = 10.f;
const int32_t INST_COUNT = 4;

const std::array<float3, INST_COUNT> g_diffuse_colors =
    {{{0.70f, 0.70f, 0.70f},
      {0.80f, 0.80f, 0.80f},
      {0.90f, 0.90f, 0.90f},
      {1.00f, 1.00f, 1.00f}}};

struct Instance
{
    float m[12];
};

const std::array<Instance, INST_COUNT> g_instances =
    {{{{1, 0, 0, -4.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
      {{1, 0, 0, -1.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
      {{1, 0, 0, 1.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
      {{1, 0, 0, 4.5f, 0, 1, 0, 0, 0, 0, 1, 0}}}};

dynamicGeometry::dynamicGeometry()
{
}

dynamicGeometry::~dynamicGeometry()
{
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(hit_group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context));

    CUDA_CHECK(cudaFree((void*)sbt.raygenRecord));
    CUDA_CHECK(cudaFree((void*)sbt.missRecordBase));
    CUDA_CHECK(cudaFree((void*)sbt.hitgroupRecordBase));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_vertices)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_static_gas_output_buffer)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_deforming_gas_output_buffer)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_exploding_gas_output_buffer)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_instances)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ias_output_buffer)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));

}
    void dynamicGeometry::initLaunchParams()
    {
        params.frame_buffer = nullptr;
        params.subframe_index = 0u;

        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(Params)));
    }

    void dynamicGeometry::initCameraState()
    {
        camera.setEye(make_float3(0.f, 1.f, -20.f));
        camera.setLookat(make_float3(0, 0, 0));
        camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
        camera.setFovY(35.0f);
        camera_changed = true;
    }
    void dynamicGeometry::createContext()
    {
        CUDA_CHECK(cudaFree(0));
        CUcontext cu_ctx = 0;

        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &optixLogCallback;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optix_context));
    }

    void dynamicGeometry::createModule(std::string ptx_filename)
    {
        std::ifstream ptx_in(ptx_filename);
        if (!ptx_in)
        {
            std::cerr << "ERROR: readPTX() Failed to open file" << ptx_filename << std::endl;
            return;
        }

        std::string ptx = std::string((std::istreambuf_iterator<char>(ptx_in)),
                                      std::istreambuf_iterator<char>());
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 3;
        pipeline_compile_options.numAttributeValues = 2;

        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &module_compile_options, &pipeline_compile_options,
                                             ptx.c_str(), ptx.size(), nullptr, nullptr, &module));
    }

    void dynamicGeometry::createProgramGroups()
    {
        OptixProgramGroupOptions program_group_options = {};

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &raygen_prog_group_desc, 1,
                                            &program_group_options, nullptr, nullptr, &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &miss_prog_group_desc, 1,
                                            &program_group_options, nullptr, nullptr, &miss_group));

        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &hit_prog_group_desc, 1,
                                            &program_group_options, nullptr, nullptr, &hit_group));
    }

    void dynamicGeometry::createPipeline()
    {
        OptixProgramGroup program_groups[] = {raygen_prog_group,
                                              miss_group,
                                              hit_group};

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = 1;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OPTIX_CHECK(optixPipelineCreate(optix_context, &pipeline_compile_options, &pipeline_link_options,
                                        program_groups, sizeof(program_groups) / sizeof(program_groups[0]), nullptr, nullptr, &pipeline));

        OptixStackSizes stack_sizes = {};
        OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_prog_group, &stack_sizes));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(miss_group, &stack_sizes));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(hit_group, &stack_sizes));

        uint32_t max_trace_depth = 1;
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 0;
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;

        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, max_cc_depth,
                                               max_dc_depth, &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));

        const uint32_t max_traversable_graph_depth = 2;
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              max_traversable_graph_depth));
    }

    void dynamicGeometry::createSBT()
    {
        CUdeviceptr d_raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc((void **)&d_raygen_record, raygen_record_size));

        RayGenSbtRecord rg_sbt = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy((void *)d_raygen_record, &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr d_miss_records;
        const size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc((void **)&d_miss_records, miss_record_size));

        MissSbtRecord ms_sbt[1];
        CUDA_CHECK(cudaMalloc((void **)&d_miss_records, miss_record_size));
        ms_sbt[0].data.bg_color = make_float4(0.0f);

        CUDA_CHECK(cudaMemcpy((void *)d_miss_records, ms_sbt,
                              miss_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr d_hitgroup_records;
        const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc((void **)&d_hitgroup_records, hitgroup_record_size * g_instances.size()));
        std::vector<HitGroupSbtRecord> hitgroup_records(g_instances.size());
        for (int i = 0; i < g_instances.size(); i++)
        {
            const int sbt_idx = i;
            OPTIX_CHECK(optixSbtRecordPackHeader(hit_group, &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.color = g_diffuse_colors[i];
        }

        CUDA_CHECK(cudaMemcpy((void *)d_hitgroup_records, hitgroup_records.data(),
                              hitgroup_record_size * hitgroup_records.size(), cudaMemcpyHostToDevice));

        sbt.raygenRecord = d_raygen_record;
        sbt.missRecordBase = d_miss_records;
        sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hitgroup_records;
        sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
        sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
    }
