#include "kernels/reshape_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reshape_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {3, 3, 6, 8, 0, -0.5},
                    {0, -1, 0.75, 1, 9, 2},
                },
                {
                    {1, 2, -5, 1, 5, 1},
                    {12, 0, 12, 2, 9, 0},
                },
            },
        },
        cpu_allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                8_p,
                1_p,
                3_p,
            },
        },
        DataType::FLOAT,
    };

    GenericTensorAccessorW result =
        create_random_filled_accessor_w(output_shape, cpu_allocator);

    reshape_cpu_forward_kernel(
        /*input=*/input,
        /*output=*/result);

    GenericTensorAccessorW correct = create_3d_accessor_w_with_contents<float>(
        {
            {
                {3, 3, 6},
            },
            {
                {8, 0, -0.5},
            },
            {
                {0, -1, 0.75},
            },
            {
                {1, 9, 2},
            },
            {
                {1, 2, -5},
            },
            {
                {1, 5, 1},
            },
            {
                {12, 0, 12},
            },
            {
                {2, 9, 0},
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("reshape_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR output_grad =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {5, 1, 1},
                },
                {
                    {2, 0.5, -0.3},
                },
                {
                    {-1, 6, 0},
                },
                {
                    {9, -10, 8},
                },
                {
                    {6, 3, -1},
                },
                {
                    {-1, 0, 0},
                },
                {
                    {0.75, 1, 5},
                },
                {
                    {0, 9, -9},
                },
            },
            cpu_allocator);

    TensorShape input_grad_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                2_p,
                2_p,
                6_p,
            },
        },
        DataType::FLOAT,
    };

    GenericTensorAccessorW result =
        create_random_filled_accessor_w(input_grad_shape, cpu_allocator);

    reshape_cpu_backward_kernel(
        /*output_grad=*/output_grad,
        /*input_grad=*/result);

    GenericTensorAccessorW correct = create_4d_accessor_w_with_contents<float>(
        {
            {
                {
                    {5, 1, 1, 2, 0.5, -0.3},
                    {-1, 6, 0, 9, -10, 8},
                },
                {
                    {6, 3, -1, -1, 0, 0},
                    {0.75, 1, 5, 0, 9, -9},
                },
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}
