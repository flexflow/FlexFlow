#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "kernels/cast_kernels_cpu.h"
#include "test_utils.h"
#include <type_traits>

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Cast Forward and Backward Kernels") {
    ManagedStream mStream = get_managed_stream();

    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape =
        make_float_tensor_shape_from_legion_dims({100, 100});
    TensorShape output_shape =
        make_double_tensor_shape_from_legion_dims({100, 100});

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));

      Kernels::Cast::forward_kernel(mStream.stream,
                                    input_accessor,
                                    output_accessor,
                                    DataType::FLOAT,
                                    DataType::DOUBLE);

      std::vector<double> host_double_data =
          load_data_to_host_from_device<double>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(host_double_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW grad_output_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Cast::backward_kernel(
          mStream.stream,
          read_only_accessor_from_write_accessor(output_accessor),
          grad_output_accessor,
          DataType::DOUBLE,
          DataType::FLOAT);

      std::vector<float> host_grad_float_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(grad_output_accessor));
      CHECK(contains_non_zero(host_grad_float_data));
    }
  }
}
