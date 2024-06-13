#include "doctest/doctest.h"
#include "kernels/partition_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Partition Forward and Backward") {
    const std::size_t num_elements = 100;
    const std::size_t num_replicas = 10;

    ArrayShape shape{std::vector<size_t>{num_elements}};

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&input_data, num_elements, 1.0f);

    RepartitionPerDeviceState state =
        Kernels::Repartition::init_kernel(handle, DataType::FLOAT);

    SUBCASE("Test forward partition kernel") {
      fillDeviceDataNum(&output_data, num_elements, 0.0f);

      GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
      GenericTensorAccessorW forward_output_accessor{
          DataType::FLOAT, shape, output_data};

      Kernels::Repartition::forward_kernel(
          stream, state, input_accessor, forward_output_accessor);

      std::vector<float> check_output_data(num_elements);
      checkCUDA(cudaMemcpy(check_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(check_output_data[i] == 1.0f);
      }
    }

    SUBCASE("Test backward partition kernel") {
      float *grad_data = static_cast<float *>(
          allocator.allocate(num_elements * sizeof(float)));
      fillDeviceDataNum(&grad_data, num_elements, 1.0f);

      GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
      GenericTensorAccessorW forward_output_accessor{
          DataType::FLOAT, shape, output_data};
      Kernels::Repartition::forward_kernel(
          stream, state, input_accessor, forward_output_accessor);

      GenericTensorAccessorR grad_accessor{DataType::FLOAT, shape, grad_data};
      Kernels::Repartition::backward_kernel(
          stream, state, forward_output_accessor, grad_accessor);

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
