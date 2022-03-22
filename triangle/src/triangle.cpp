#include "triangle.hpp"

Triangle::Triangle()
{
    m_width = 1024;
    m_height = 768;
}

Triangle::~Triangle()
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));

    OPTIX_CHECK(optixDeviceContextDestroy(optix_context));
}
void Triangle::configCamera()
{
    camera.setEye({0.0f, 0.0f, 2.0f});
    camera.setLookat({0.0f, 0.0f, 0.0f});
    camera.setUp({0.0f, 1.0f, 3.0f});
    camera.setFovY(45.0f);
    camera.setAspectRatio((float)m_width / (float)m_height);
}

void Triangle::createContext()
{
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 4;
    // Associate a CUDA context (and therefore a specific GPU) with this device context
    CUcontext cuCtx = 0; // zero means take the current context

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optix_context));
}

void Triangle::buildGAS()
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const std::array<float3, 3> vertices =
        {{{-0.5f, -0.5f, 0.0f},
          {0.5f, -0.5f, 0.0f},
          {0.0f, 0.5f, 0.0f}}};

    const size_t vertices_size = sizeof(float3) * vertices.size();

    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice));

    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
                          gas_buffer_sizes.tempSizeInBytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer),
                          gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(optix_context,
                                0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
}

void Triangle::createModule(const std::string ptx_filename)
{
    std::ifstream ptx_in(ptx_filename);
    if (!ptx_in)
    {
        std::cerr << "ERROR: readPTX() Failed to open file " << ptx_filename
                  << std::endl;
        return;
    }
    std::string ptx = std::string((std::istreambuf_iterator<char>(ptx_in)),
                                  std::istreambuf_iterator<char>());

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 3;

    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    size_t inputSize = 0;
    // const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, ptx_filename.c_str(), inputSize);
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &module_compile_options, &pipeline_compile_options,
                                         ptx.c_str(), ptx.size(), nullptr, nullptr, &module));
}

void Triangle::createProgramGroups()
{
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &hitgroup_prog_group));
}

void Triangle::createPipeline()
{
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    OPTIX_CHECK(optixPipelineCreate(
        optix_context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        nullptr,
        nullptr,
        &pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));
}

void Triangle::buildSBT()
{
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.3f, 0.1f, 0.2f};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;
}

void Triangle::launch()
{
    sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, m_width, m_height);

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    configCamera();

    Params params;
    params.image = output_buffer.map();
    params.image_width = m_width;
    params.image_height = m_height;
    params.handle = gas_handle;
    params.cam_eye = camera.eye();
    camera.UVWFrame(params.cam_u, params.cam_v, params.cam_w);

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_param),
        &params, sizeof(params),
        cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, m_width, m_height, /*depth=*/1));
    CUDA_SYNC_CHECK();

    output_buffer.unmap();

    sutil::ImageBuffer buffer;
    buffer.data = output_buffer.getHostPointer();
    buffer.width = m_width;
    buffer.height = m_height;
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    cv::Mat img(output_buffer.height(), output_buffer.width(), CV_8UC3);
    getMatImage(output_buffer, img);
    cv::imshow("img", img);
    cv::waitKey();
}

void Triangle::init()
{
    createContext();
    buildGAS();
    createModule("triangle/ptx/triangle.ptx");
    createProgramGroups();
    createPipeline();
    buildSBT();
}