#include "doctest/doctest.h"
#include "kernels/reduction_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_elements = 10;
    std::size_t num_replicas = 10;
    std::size_t total_elements = num_elements * num_replicas;

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{num_elements},
    };
    ArrayShape expanded_shape = ArrayShape{
        std::vector<size_t>{total_elements},
    };

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    GenericTensorAccessorW *output_accessor_ptr;
    SUBCASE("Test Reduction Forward") {
      float *input_data, *output_data;
      std::vector<float **> ptrs = {&input_data, &output_data};
      std::vector<size_t> sizes = {total_elements, num_elements};
      allocate_ptrs(ptrs, sizes, allocator);

      GenericTensorAccessorR input_accessor{
          DataType::FLOAT, expanded_shape, input_data};
      GenericTensorAccessorW output_accessor{
          DataType::FLOAT, shape, output_data};
      output_accessor_ptr = &output_accessor;

      randomFillDeviceData(&input_data, total_elements);

      Kernels::Reduction::forward_kernel(
          stream, input_accessor, output_accessor, num_replicas);
    }

    SUBCASE("Test Reduction Backward") {
      float *grad_input_data = static_cast<float *>(
          allocator.allocate(total_elements * sizeof(float)));
      fillDeviceDataNum(&grad_input_data, total_elements, 1.0f);
      GenericTensorAccessorR grad_accessor{
          DataType::FLOAT, shape, grad_input_data};

      Kernels::Reduction::backward_kernel(
          stream, *output_accessor_ptr, grad_accessor);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
