cmake_minimum_required(VERSION 3.19)
project(TrinityVisu LANGUAGES CXX CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FindPkgConfig)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio highgui imgproc)
find_package(Freetype REQUIRED)

# TensorRT setup (installed at /usr/src/tensorrt or /usr)
set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT root directory")
find_library(NVINFER_LIBRARY NAMES nvinfer PATHS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser PATHS ${TENSORRT_ROOT}/lib)
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin PATHS ${TENSORRT_ROOT}/lib)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS ${TENSORRT_ROOT}/include)

if(NOT NVINFER_LIBRARY OR NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT to the TensorRT installation directory.")
endif()

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/corekit)

add_library(trinity INTERFACE)

add_dependencies(trinity 
    corekit
)

target_include_directories(trinity INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(trinity INTERFACE
    corekit
    ${OpenCV_LIBS}
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
)

###################################################################

add_library(cuda_ops STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/core.cu   
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/image.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/draw.cu
)

target_include_directories(cuda_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${FREETYPE_INCLUDE_DIRS}
)

target_link_libraries(cuda_ops PUBLIC
    ${FREETYPE_LIBRARIES}
)

add_executable(visu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(visu PUBLIC 
    trinity
    cuda_ops
)

# NVCC struggles with C++20 ranges used by nlohmann::json; disable ranges in CUDA builds.
target_compile_definitions(visu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:JSON_HAS_RANGES=0>
)