#include "doctest/doctest.h"
#include "test_utils.h"

#include <random>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test CUDA") {
    int deviceCount = 0;

    cudaError_t device_error = cudaGetDeviceCount(&deviceCount);
    CHECK(device_error == cudaSuccess);
    CHECK(deviceCount > 0);

    int driverVersion = 0;
    cudaError_t driver_error = cudaDriverGetVersion(&driverVersion);
    CHECK(driver_error == cudaSuccess);
    CHECK(driverVersion > 0);

    int runtimeVersion = 0;
    cudaError_t runtime_error = cudaRuntimeGetVersion(&runtimeVersion);
    CHECK(runtime_error == cudaSuccess);
    CHECK(runtimeVersion > 0);

    if (device_error == cudaSuccess) {
      void *ptr;
      checkCUDA(cudaMalloc(&ptr, 1));
      checkCUDA(cudaFree(ptr));
    }
  }
}
} // namespace FlexFlow
