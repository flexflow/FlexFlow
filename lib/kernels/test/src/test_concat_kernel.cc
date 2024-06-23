#include "doctest/doctest.h"
#include "kernels/concat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    size_t num_inputs = 3;
    size_t size_per_input = 100;
    ff_dim_t concat_axis = ff_dim_t(0);

    ManagedStream mStream = get_managed_stream();

    TensorShape input_shape =
        make_float_tensor_shape_from_legion_dims({size_per_input});
    TensorShape output_shape =
        make_float_tensor_shape_from_legion_dims({size_per_input, num_inputs});

    Allocator allocator = get_local_memory_allocator();

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("forward_kernel") {
      std::vector<GenericTensorAccessorR> input_accessors =
          repeat(num_inputs, [&]() {
            return read_only_accessor_from_write_accessor(
                create_random_filled_accessor_w(input_shape, allocator));
          });

      Kernels::Concat::forward_kernel(
          mStream.stream, output_accessor, input_accessors, concat_axis);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      std::vector<GenericTensorAccessorW> grad_input_accessors = repeat(
          num_inputs, [&]() { return allocator.allocate_tensor(input_shape); });
      GenericTensorAccessorR grad_output_accessor =
          read_only_accessor_from_write_accessor(output_accessor);

      Kernels::Concat::backward_kernel(mStream.stream,
                                       grad_output_accessor,
                                       grad_input_accessors,
                                       concat_axis);
    }
  }
}
