cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(mrhash LANGUAGES C CXX CUDA)

# Allow user to override build type, default to Release for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

# opencv
find_package(OpenCV REQUIRED)
message(STATUS "${PROJECT_NAME}: OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# cuda
find_file(CUDA_HEADER cuda_runtime.h PATHS /usr/include/ /usr/local/cuda/include/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${CUDA_HEADER}")

# Note: Suppressing warning 20012/20014 for float*/Eigen::Matrix type compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012 -diag-suppress 20014")

# Auto-detect CUDA architecture
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

if(NOT CUDA_ARCH_LIST)
  # Fallback to common architectures if auto-detection fails
  set(CUDA_ARCH_LIST "75;80;86;89" CACHE STRING "CUDA architectures" FORCE)
  message(WARNING "Could not auto-detect CUDA architecture, using fallback: ${CUDA_ARCH_LIST}")
else()
  message(STATUS "Detected CUDA architectures: ${CUDA_ARCH_LIST}")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)
message(STATUS "Eigen Include Directories: ${eigen_SOURCE_DIR}")

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.7.0
)
FetchContent_MakeAvailable(nanobind)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

# gtests
find_package(GTest REQUIRED)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  # GIT_TAG yaml-cpp-0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

# Libigl
# list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" ON)
# include(libigl)
# igl_include(core)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Torch setup
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Define the absolute path to LibTorch
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/third_party/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${LIBTORCH_DIR}/lib")

add_subdirectory(third_party)

set(GEOM_LIB_NAME "libgeometric")
set(GS_LIB_NAME "libgsmodel")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(apps)
