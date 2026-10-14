#include "kernels/transpose_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transpose_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = TransposeAttrs{
        /*permutation=*/TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                {ff_dim_t{0_n}, ff_dim_t{3_n}},
                {ff_dim_t{1_n}, ff_dim_t{0_n}},
                {ff_dim_t{2_n}, ff_dim_t{2_n}},
                {ff_dim_t{3_n}, ff_dim_t{1_n}},
            },
        },
    };

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {3, 3, 6},
                    {0, -1, 0.75},
                },
                {
                    {1, 2, -5},
                    {12, 0, 12},
                },
                {
                    {8, 0, -0.5},
                    {1, 9, 2},
                },
                {
                    {1, 5, 1},
                    {2, 9, 0},
                },
            },
        },
        cpu_allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                3_p,
                1_p,
                2_p,
                4_p,
            },
        },
        DataType::FLOAT,
    };

    GenericTensorAccessorW result =
        create_random_filled_accessor_w(output_shape, cpu_allocator);

    transpose_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/result);

    GenericTensorAccessorR correct = create_4d_accessor_w_with_contents<float>(
        {
            {
                {
                    {3, 1, 8, 1},
                    {0, 12, 1, 2},
                },
            },
            {
                {
                    {3, 2, 0, 5},
                    {-1, 0, 9, 9},
                },
            },
            {
                {
                    {6, -5, -0.5, 1},
                    {0.75, 12, 2, 0},
                },
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("transpose_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = TransposeAttrs{
        /*permutation=*/TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                {ff_dim_t{0_n}, ff_dim_t{3_n}},
                {ff_dim_t{1_n}, ff_dim_t{0_n}},
                {ff_dim_t{2_n}, ff_dim_t{2_n}},
                {ff_dim_t{3_n}, ff_dim_t{1_n}},
            },
        },
    };

    GenericTensorAccessorR out_grad = create_4d_accessor_w_with_contents<float>(
        {
            {
                {
                    {1, 2, 9, -1},
                    {2, 3, 2, 4},
                },
            },
            {
                {
                    {0.1, 6, 2, 1},
                    {-1, 0, 0, 1},
                },
            },
            {
                {
                    {2, -5, -0.5, 1},
                    {0, 12, 0, 12},
                },
            },
        },
        cpu_allocator);

    TensorShape input_grad_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                4_p,
                2_p,
                3_p,
            },
        },
        DataType::FLOAT,
    };

    GenericTensorAccessorW result =
        create_random_filled_accessor_w(input_grad_shape, cpu_allocator);

    transpose_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*out_grad=*/out_grad,
        /*in_grad=*/result);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 0.1, 2},
                    {2, -1, 0},
                },
                {
                    {2, 6, -5},
                    {3, 0, 12},
                },
                {
                    {9, 2, -0.5},
                    {2, 0, 0},
                },
                {
                    {-1, 1, 1},
                    {4, 1, 12},
                },
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}
