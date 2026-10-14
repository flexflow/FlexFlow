#include "kernels/linear_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("linear_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    LinearAttrs attrs = LinearAttrs{
        /*out_channels=*/2_p,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 1, 5},
            {1, 2, -2},
            {8, 0.5, -3},
        },
        cpu_allocator);

    GenericTensorAccessorR projection =
        create_2d_accessor_r_with_contents<float>(
            {
                {1.0f, 2.0f, 1.5f},
                {0.5f, 4.0f, -1.0f},
            },
            cpu_allocator);

    GenericTensorAccessorR bias =
        create_1d_accessor_r_with_contents<float>({3.0, -1.0}, cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{4_p, attrs.out_channels}},
            DataType::FLOAT,
        },
        cpu_allocator);

    linear_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/result,
        /*projection=*/projection,
        /*bias=*/bias);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {21.0f, 6.5f},
            {14.5f, -1.0f},
            {5.0f, 9.5f},
            {7.5f, 8.0f},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("linear_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    LinearAttrs attrs = LinearAttrs{
        /*out_channels=*/2_p,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 1, 5},
            {1, 2, -2},
            {8, 0.5, -3},
        },
        cpu_allocator);

    GenericTensorAccessorW input_grad =
        cpu_allocator.allocate_tensor(get_tensor_shape_for_accessor_r(input));

    GenericTensorAccessorR projection =
        create_2d_accessor_r_with_contents<float>(
            {
                {1.0f, 2.0f, 1.5f},
                {0.5f, 4.0f, -1.0f},
            },
            cpu_allocator);

    GenericTensorAccessorW projection_grad = cpu_allocator.allocate_tensor(
        get_tensor_shape_for_accessor_r(projection));

    GenericTensorAccessorR bias =
        create_1d_accessor_r_with_contents<float>({3.0, -1.0}, cpu_allocator);

    GenericTensorAccessorW bias_grad =
        cpu_allocator.allocate_tensor(get_tensor_shape_for_accessor_r(bias));

    GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
        {
            {21.0f, 6.5f},
            {14.5f, -1.0f},
            {5.0f, 9.5f},
            {7.5f, 8.0f},
        },
        cpu_allocator);

    GenericTensorAccessorR output_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {1.0f, -0.5f},
                {2.0f, -2.0f},
                {1.0f, 9.0f},
                {-3.5f, 1.0f},
            },
            cpu_allocator);

    linear_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*projection=*/projection,
        /*projection_grad=*/projection_grad,
        /*bias_grad=*/bias_grad);

    GenericTensorAccessorR correct_input_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {0.75f, 0.0f, 2.0f},
                {1.0f, -4.0f, 5.0f},
                {5.5f, 38.0f, -7.5f},
                {-3.0f, -3.0f, -6.25f},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_projection_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {-20.0f, 5.25f, 24.5f},
                {11.5f, 15.0f, -34.0f},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_bias_grad =
        create_1d_accessor_r_with_contents<float>(
            {
                1.0f + 2.0f + 1.0f + -3.5f,
                -0.5f + -2.0f + 9.0f + 1.0f,
            },
            cpu_allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct_input_grad),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));

    CHECK_MESSAGE(accessors_are_equal(projection_grad, correct_projection_grad),
                  check_kv("projection_grad",
                           format_accessor_w_contents(projection_grad)));

    CHECK_MESSAGE(accessors_are_equal(bias_grad, correct_bias_grad),
                  check_kv("bias_grad", format_accessor_w_contents(bias_grad)));
  }
}
