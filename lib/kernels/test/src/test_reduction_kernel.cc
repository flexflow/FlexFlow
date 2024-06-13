#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/reduction_kernels.h"
#include "test_utils.h"
#include <algorithm>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   std::vector<size_t> const &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_elements = 10;
    std::size_t num_replicas = 10;
    std::size_t total_elements = num_elements * num_replicas;
    std::size_t dims[] = {num_elements};
    std::size_t expanded_dims[] = {total_elements};
    DataType dtype = DataType::FLOAT;

    ArrayShape shape(dims, 1);
    ArrayShape expanded_shape(expanded_dims, 1);

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {total_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    const GenericTensorAccessorR input_accessor{
        dtype, expanded_shape, input_data};
    const GenericTensorAccessorW output_accessor{dtype, shape, output_data};

    randomFillDeviceData(&input_data, total_elements);

    Kernels::Reduction::forward_kernel(
        stream, input_accessor, output_accessor, num_replicas);

    float *grad_input_data = static_cast<float *>(
        allocator.allocate(total_elements * sizeof(float)));
    fillDeviceDataNum(&grad_input_data, total_elements, 1.0f);
    const GenericTensorAccessorR grad_accessor{dtype, shape, grad_input_data};

    Kernels::Reduction::backward_kernel(stream, output_accessor, grad_accessor);
    checkCUDA(cudaStreamDestroy(stream));
  }
}
