#pragma once

#include <cuda_runtime.h>

namespace cuda {

void check(cudaError_t result, const char* context);

} // namespace cuda
