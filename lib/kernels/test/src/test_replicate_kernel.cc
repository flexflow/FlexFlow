#include "doctest/doctest.h"
#include "kernels/replicate_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Replicate Kernel") {
    std::size_t num_elements = 100;
    std::size_t num_replicas = 10;

    TensorShape shape = make_float_tensor_shape_w_legion_dims({num_elements});

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          create_filled_accessor_w(shape, allocator, 0.0f);

      Kernels::Replicate::forward_kernel(
          stream, input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(1.0f == check_output_data[i]);
      }

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR replicated_accessor =
            read_only_accessor_from_write_accessor(
                create_filled_accessor_w(shape, allocator, num_replicas));
        GenericTensorAccessorW aggregated_accessor =
            create_filled_accessor_w(shape, allocator, 0.0f);

        Kernels::Replicate::backward_kernel(
            stream, aggregated_accessor, replicated_accessor, num_replicas);

        std::vector<float> check_aggregated_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(aggregated_accessor));
        REQUIRE(contains_non_zero(check_aggregated_data));
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
