#include "doctest/doctest.h"
#include "kernels/partition_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Partition Forward and Backward") {
    const std::size_t num_elements = 100;
    const std::size_t num_replicas = 10;

    TensorShape shape = get_float_tensor_shape({num_elements});

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    RepartitionPerDeviceState state =
        Kernels::Repartition::init_kernel(handle, DataType::FLOAT);

    SUBCASE("Test forward partition kernel") {
      GenericTensorAccessorR input_accessor =
          makeReadOnlyAccessor(getFilledAccessorW(shape, allocator, 1.0f));
      GenericTensorAccessorW forward_output_accessor =
          getFilledAccessorW(shape, allocator, 0.0f);

      Kernels::Repartition::forward_kernel(
          stream, state, input_accessor, forward_output_accessor);

      std::vector<float> check_output_data =
          fill_host_data<float>(forward_output_accessor.ptr, num_elements);

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(check_output_data[i] == 1.0f);
      }

      SUBCASE("Test backward partition kernel") {
        GenericTensorAccessorR grad_accessor =
            makeReadOnlyAccessor(getFilledAccessorW(shape, allocator, 1.0f));

        Kernels::Repartition::backward_kernel(
            stream, state, forward_output_accessor, grad_accessor);

        std::vector<float> host_grad_input_data =
            fill_host_data<float>(forward_output_accessor.ptr, num_elements);

        for (std::size_t i = 0; i < num_elements; ++i) {
          CHECK(host_grad_input_data[i] == 2.0f);
        }
      }
    }

    cleanup_test(stream, handle); 
  }
}
