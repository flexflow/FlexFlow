#include "doctest/doctest.h"
#include "kernels/concat_kernels.h"
#include "test_utils.h"
#include "utils/containers/repeat.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    size_t num_inputs = 2;
    size_t size_per_input = 10;
    ff_dim_t concat_axis = ff_dim_t(1);

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({size_per_input}, DataType::FLOAT);
    TensorShape output_shape = make_tensor_shape_from_legion_dims(
        {num_inputs, size_per_input}, DataType::FLOAT);

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      std::vector<GenericTensorAccessorR> input_accessors =
          repeat(num_inputs, [&]() {
            return create_random_filled_accessor_r(input_shape, allocator);
          });
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Concat::forward_kernel(managed_stream.raw_stream(),
                                      output_accessor,
                                      input_accessors,
                                      concat_axis);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      std::vector<GenericTensorAccessorW> input_grad_accessors = repeat(
          num_inputs, [&]() { return allocator.allocate_tensor(input_shape); });

      Kernels::Concat::backward_kernel(managed_stream.raw_stream(),
                                       output_grad_accessor,
                                       input_grad_accessors,
                                       concat_axis);
    }
  }
}
