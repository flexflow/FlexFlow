#include "doctest/doctest.h"
#include "kernels/gather_kernels.h"
#include "kernels/local_allocator.h"
#include "test_utils.h"
#include <random>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   const std::vector<size_t> &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    size_t num_elements = 100;
    size_t output_size = 50;
    size_t stride = 1;
    size_t input_dim_size = num_elements;
    size_t output_dim_size = output_size;

    size_t dims[] = {num_elements};
    ArrayShape shape(dims, 1);

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Allocator allocator = get_local_memory_allocator();

    float *device_input, *device_output, *device_indices;
    std::vector<float **> ptrs = {&device_input, &device_output,
                                  &device_indices};
    std::vector<size_t> sizes = {num_elements, output_size, output_size};
    allocate_ptrs(ptrs, sizes, allocator);

    const GenericTensorAccessorW device_output_accessor{DataType::FLOAT, shape,
                                                        device_input};
    const GenericTensorAccessorR device_input_accessor{DataType::FLOAT, shape,
                                                       device_input};
    const GenericTensorAccessorR device_indices_accessor{
        DataType::FLOAT, ArrayShape({output_size}), device_indices};

    randomFillDeviceData(&device_input, num_elements);
    randomFillDeviceData(&device_indices, output_size);

    GatherPerDeviceState state = {2, DataType::FLOAT};
    Kernels::Gather::forward_kernel(
        stream, state, device_input_accessor, device_indices_accessor,
        device_output_accessor, stride, input_dim_size, output_dim_size);

    std::vector<float> host_output(output_size, 0.0f);
    cudaMemcpy(host_output.data(), device_output, output_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudnnDestroy(handle.dnn);
    cublasDestroy(handle.blas);
  }
}
