#include "internal/test_utils.h"
#include "kernels/flat_kernels_gpu.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    Allocator allocator = create_local_cuda_memory_allocator();

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(create_filled_accessor_w(
            input_shape, allocator, make_float_data_type_value(2)));

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      flat_gpu_forward_kernel(
          managed_stream.raw_stream(),
          input_accessor.get_float_ptr(),
          output_accessor.get_float_ptr(),
          get_num_elements(input_shape.dims).size_t_from_positive_int(),
          size_of_datatype(input_shape.data_type).size_t_from_positive_int());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad_accessor = create_filled_accessor_r(
          output_shape, allocator, make_float_data_type_value(0));
      GenericTensorAccessorW input_grad_accessor = create_filled_accessor_w(
          input_shape, allocator, make_float_data_type_value(1));

      flat_gpu_backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor.get_float_ptr(),
          input_grad_accessor.get_float_ptr(),
          get_num_elements(input_shape.dims).size_t_from_positive_int());

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
