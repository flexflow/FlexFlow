#include "doctest/doctest.h"
#include "kernels/reshape_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
    const std::size_t num_elements = 100;
    ArrayShape shape = ArrayShape{
        std::vector<size_t>{num_elements},
    };

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test Reshape Forward") {
      float *input_data, *output_data;
      std::vector<float **> ptrs = {&input_data, &output_data};
      std::vector<size_t> sizes = {num_elements, num_elements};
      allocate_ptrs(ptrs, sizes, allocator);

      fillDeviceDataNum(&input_data, num_elements, 1.0f);

      GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
      GenericTensorAccessorW forward_output_accessor{
          DataType::FLOAT, shape, output_data};

      ReshapePerDeviceState state =
          Kernels::Reshape::init_kernel(DataType::FLOAT);

      Kernels::Reshape::forward_kernel(
          stream, state, input_accessor, forward_output_accessor);

      std::vector<float> check_output_data(num_elements);
      checkCUDA(cudaMemcpy(check_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(1.0f == check_output_data[i]);
      }
    }

    SUBCASE("Test Reshape Kernel Backward") {
      float *output_data, *grad_data;
      std::vector<float **> ptrs = {&output_data, &grad_data};
      std::vector<size_t> sizes = {num_elements, num_elements};
      allocate_ptrs(ptrs, sizes, allocator);

      fillDeviceDataNum(&output_data, num_elements, 1.0f);
      fillDeviceDataNum(&grad_data, num_elements, 1.0f);

      GenericTensorAccessorR grad_accessor{DataType::FLOAT, shape, grad_data};
      GenericTensorAccessorW backward_output_accessor{
          DataType::FLOAT, shape, output_data};

      ReshapePerDeviceState state =
          Kernels::Reshape::init_kernel(DataType::FLOAT);

      Kernels::Reshape::backward_kernel(
          stream, state, backward_output_accessor, grad_accessor);

      std::vector<float> host_grad_input_data(num_elements);
      checkCUDA(cudaMemcpy(host_grad_input_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        CHECK(host_grad_input_data[i] == 2.0f);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
