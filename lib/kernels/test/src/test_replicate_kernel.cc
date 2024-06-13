#include "doctest/doctest.h"
#include "kernels/replicate_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Replicate Kernel") {
    std::size_t num_elements = 100;
    std::size_t num_replicas =
        10; // Assuming you have a certain number of replicas
    ArrayShape shape = ArrayShape{
        std::vector<size_t>{num_elements},
    };

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&input_data, num_elements, 1.0f);

    SUBCASE("Test Replicate Forward") {
      GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};
      GenericTensorAccessorW forward_output_accessor{
          DataType::FLOAT, shape, output_data};

      Kernels::Replicate::forward_kernel(
          stream, input_accessor, forward_output_accessor);

      std::vector<float> check_output_data(num_elements);
      checkCUDA(cudaMemcpy(check_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(1.0f == check_output_data[i]);
      }
    }

    SUBCASE("Test Replicate Backward") {
      float *replicated_data, *aggregated_data;
      ptrs = {&replicated_data, &aggregated_data};
      sizes = {num_elements * num_replicas, num_elements};
      allocate_ptrs(ptrs, sizes, allocator);

      for (size_t i = 0; i < num_replicas; ++i) {
        checkCUDA(cudaMemcpy(replicated_data + i * num_elements,
                             input_data,
                             num_elements * sizeof(float),
                             cudaMemcpyDeviceToDevice));
      }

      GenericTensorAccessorR replicated_accessor{
          DataType::FLOAT,
          ArrayShape{std::vector<size_t>{num_elements * num_replicas}},
          replicated_data};
      GenericTensorAccessorW aggregated_accessor{
          DataType::FLOAT, shape, aggregated_data};

      Kernels::Replicate::backward_kernel(
          stream, aggregated_accessor, replicated_accessor, num_replicas);

      std::vector<float> check_aggregated_data(num_elements);
      checkCUDA(cudaMemcpy(check_aggregated_data.data(),
                           aggregated_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(check_aggregated_data[i] == num_replicas);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
