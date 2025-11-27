FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	1.0.2 
)

FetchContent_MakeAvailable(glm)

add_library(${GS_LIB_NAME} STATIC
    quad_tree.cu
    gaussian.cu
    rasterizer.cu
    rasterize_points.cu
    cuda_rasterizer/src/backward.cu
    cuda_rasterizer/src/forward.cu
    cuda_rasterizer/src/rasterizer_impl.cu)

target_include_directories(${GS_LIB_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src/sdf
    cuda_rasterizer/include
    ${eigen_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/tinyply/source)

set_target_properties(${GS_LIB_NAME} PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    UNITY_BUILD ON
)

target_compile_options(${GS_LIB_NAME} PRIVATE -O3 -use_fast_math -Xcompiler -Ofast -lineinfo $<$<COMPILE_LANGUAGE:CUDA>: -g -lineinfo >)
find_library(NVML_LIBRARY nvidia-ml PATHS /usr/lib/x86_64-linux-gnu)
target_link_libraries(${GS_LIB_NAME} PUBLIC
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    tinyply
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    glm::glm
)
