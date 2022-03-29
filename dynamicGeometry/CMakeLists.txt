project(dynamicGeometry LANGUAGES CXX CUDA)

include_directories(./include)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/../common/*.cpp)
file(GLOB CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(${PROJECT_NAME} ${SOURCES})
set(NVCC_OPTIONS "--gpu-architecture=compute_61;--use_fast_math;--relocatable-device-code=true;--generate-line-info;-Wno-deprecated-gpu-targets;-Wno-deprecated-declarations;-I${OptiX_INCLUDE};-I${OptiX_INSTALL_DIR}/SDK")
add_ptx_targets(${PROJECT_NAME} ${CUDA_SOURCES})

target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cuda_driver OptiX7)
target_link_libraries(${PROJECT_NAME} -L${OPTIX_HOME}/build/lib -lsutil_7_sdk -lglad -lglfw)
# OpenCV
find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

