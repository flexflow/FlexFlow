#include "doctest/doctest.h"
#include "kernels/gather_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    size_t num_elements = 100;
    size_t output_size = 50;

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{num_elements},
    };

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Allocator allocator = get_local_memory_allocator();

    float *device_input, *device_output, *device_indices;
    std::vector<float **> ptrs = {
        &device_input, &device_output, &device_indices};
    std::vector<size_t> sizes = {num_elements, output_size, output_size};
    allocate_ptrs(ptrs, sizes, allocator);

    SUBCASE("Test Gather Forward") {
      GenericTensorAccessorW device_output_accessor{
          DataType::FLOAT, shape, device_input};
      GenericTensorAccessorR device_input_accessor{
          DataType::FLOAT, shape, device_input};
      GenericTensorAccessorR device_indices_accessor{
          DataType::FLOAT, ArrayShape({output_size}), device_indices};

      randomFillDeviceData(&device_input, num_elements);
      randomFillDeviceData(&device_indices, output_size);

      GatherPerDeviceState state = {handle, legion_dim_t(2)};
      Kernels::Gather::forward_kernel(stream,
                                      state,
                                      device_input_accessor,
                                      device_indices_accessor,
                                      device_output_accessor);
      std::vector<float> host_output(output_size, 0.0f);
      cudaMemcpy(host_output.data(),
                 device_output,
                 output_size * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }

    SUBCASE("Test Gather Backward") {
      // Will add later
    }

    cudaStreamDestroy(stream);
    cudnnDestroy(handle.dnn);
    cublasDestroy(handle.blas);
  }
}
