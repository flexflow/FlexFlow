#include "doctest/doctest.h"
#include "kernels/reduction_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_elements = 10;
    std::size_t num_replicas = 10;
    std::size_t total_elements = num_elements * num_replicas;

    TensorShape shape = make_float_tensor_shape_w_legion_dims({num_elements});
    TensorShape expanded_shape =
        make_float_tensor_shape_w_legion_dims({total_elements});

    PerDeviceFFHandle handle = get_per_device_ff_handle();
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(expanded_shape, allocator));
      GenericTensorAccessorW output_accessor =
          create_random_filled_accessor_w(expanded_shape, allocator);

      Kernels::Reduction::forward_kernel(
          stream, input_accessor, output_accessor, num_replicas);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR grad_accessor =
            read_only_accessor_from_write_accessor(
                create_filled_accessor_w(expanded_shape, allocator, 1.0f));

        Kernels::Reduction::backward_kernel(
            stream, output_accessor, grad_accessor);

        std::vector<float> host_grad_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(output_accessor));
      }
    }

    cleanup_test(stream, handle);
  }
}
