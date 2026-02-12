cmake_minimum_required(VERSION 3.19)
project(TrinityVisu LANGUAGES CXX CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FindPkgConfig)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio highgui imgproc)

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
    trinity
    ${OpenCV_LIBS}
)

###################################################################

add_library(cuda_ops STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/core.cu   
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/image.cu
)

target_include_directories(cuda_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
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