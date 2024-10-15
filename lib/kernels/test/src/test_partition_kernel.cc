#include "doctest/doctest.h"
#include "kernels/partition_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Partition Forward and Backward") {
    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    RepartitionPerDeviceState state = Kernels::Repartition::init_kernel(
        managed_handle.raw_handle(), DataType::FLOAT);

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({10, 10}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_filled_accessor_r(input_shape, allocator, 1.0f);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Repartition::forward_kernel(
          managed_stream.raw_stream(), state, input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_filled_accessor_r(output_shape, allocator, 1.0f);
      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w(input_shape, allocator, 2.0f);

      Kernels::Repartition::backward_kernel(managed_stream.raw_stream(),
                                            state,
                                            input_grad_accessor,
                                            output_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
