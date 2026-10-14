#include "kernels/split_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("split_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 3, 2, 12, -1, 2},
            {2, 1, 5, 9, 5, 10},
        },
        cpu_allocator);

    SplitAttrs attrs = SplitAttrs{
        /*splits=*/{3_p, 1_p, 2_p},
        /*axis=*/ff_dim_t{1_n},
    };

    auto mk_output_shape = [&](positive_int axis_size) -> TensorShape {
      return TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  2_p,
                  axis_size,
              },
          },
          DataType::FLOAT,
      };
    };

    GenericTensorAccessorW output_0 =
        create_zero_filled_accessor_w(mk_output_shape(3_p), cpu_allocator);

    GenericTensorAccessorW output_1 =
        create_zero_filled_accessor_w(mk_output_shape(1_p), cpu_allocator);

    GenericTensorAccessorW output_2 =
        create_zero_filled_accessor_w(mk_output_shape(2_p), cpu_allocator);

    split_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*outputs=*/{output_0, output_1, output_2});

    GenericTensorAccessorR correct_0 =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2},
                {2, 1, 5},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_1 =
        create_2d_accessor_r_with_contents<float>(
            {
                {12},
                {9},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_2 =
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, 2},
                {5, 10},
            },
            cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(output_0, correct_0),
                  check_kv("output_0", format_accessor_w_contents(output_0)));
    CHECK_MESSAGE(accessors_are_equal(output_1, correct_1),
                  check_kv("output_1", format_accessor_w_contents(output_1)));
    CHECK_MESSAGE(accessors_are_equal(output_2, correct_2),
                  check_kv("output_2", format_accessor_w_contents(output_2)));
  }

  TEST_CASE("split_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SplitAttrs attrs = SplitAttrs{
        /*splits=*/{3_p, 1_p, 2_p},
        /*axis=*/ff_dim_t{1_n},
    };

    GenericTensorAccessorR output_grad_0 =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2},
                {2, 1, 5},
            },
            cpu_allocator);

    GenericTensorAccessorR output_grad_1 =
        create_2d_accessor_r_with_contents<float>(
            {
                {12},
                {9},
            },
            cpu_allocator);

    GenericTensorAccessorR output_grad_2 =
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, 2},
                {5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_input_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2, 12, -1, 2},
                {2, 1, 5, 9, 5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorW input_grad =
        create_zero_filled_accessor_w(correct_input_grad.shape, cpu_allocator);

    split_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grads=*/{output_grad_0, output_grad_1, output_grad_2},
        /*input_grad=*/input_grad);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct_input_grad),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}
