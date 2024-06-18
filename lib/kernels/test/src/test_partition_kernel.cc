#include "doctest/doctest.h"
#include "kernels/partition_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Partition Forward and Backward") {
    std::size_t num_elements = 100;
    std::size_t num_replicas = 10;

    TensorShape shape = make_float_tensor_shape_w_legion_dims({num_elements});

    PerDeviceFFHandle handle = get_per_device_ff_handle();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    RepartitionPerDeviceState state =
        Kernels::Repartition::init_kernel(handle, DataType::FLOAT);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));
      GenericTensorAccessorW forward_output_accessor =
          create_filled_accessor_w(shape, allocator, 0.0f);

      Kernels::Repartition::forward_kernel(
          stream, state, input_accessor, forward_output_accessor);

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(forward_output_accessor));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(check_output_data[i] == 1.0f);
      }

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR grad_accessor =
            read_only_accessor_from_write_accessor(
                create_filled_accessor_w(shape, allocator, 1.0f));

        Kernels::Repartition::backward_kernel(
            stream, state, forward_output_accessor, grad_accessor);

        std::vector<float> host_grad_input_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(
                    forward_output_accessor));

        for (std::size_t i = 0; i < num_elements; ++i) {
          CHECK(host_grad_input_data[i] == 2.0f);
        }
      }
    }

    cleanup_test(stream, handle);
  }
}
