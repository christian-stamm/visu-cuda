#include "cuda/core.hpp"

#include <cstdlib>
#include <iostream>

namespace cuda {

void checkCuda(cudaError_t result, const char* context)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << context << ": " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

} // namespace cuda
