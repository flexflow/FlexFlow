#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_accessor_binary_concat") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 1, 5},
            {1, 2, -2},
            {8, 0.5, -3},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {1, 5},
            {2, 1},
            {4, 4},
            {-3, 0},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_binary_concat(lhs, rhs, ff_dim_t{1_n}, cpu_allocator);

    GenericTensorAccessorW correct = create_2d_accessor_w_with_contents<float>(
        {
            {3, 3, 6, 1, 5},
            {2, 1, 5, 2, 1},
            {1, 2, -2, 4, 4},
            {8, 0.5, -3, -3, 0},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_elementwise_divide") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 0, 5},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 1, -10},
            {-2, 3, -2},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_elementwise_divide(lhs, rhs, cpu_allocator);

    GenericTensorAccessorW correct = create_2d_accessor_w_with_contents<float>(
        {
            {0.6f, 3.0f, -0.6f},
            {-1.0f, 0.0f, -2.5f},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_elementwise_max") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3.5, 6},
            {0, 0, -5},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 1, 10},
            {-2, 3, -2},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_elementwise_max(lhs, rhs, cpu_allocator);

    GenericTensorAccessorW correct = create_2d_accessor_w_with_contents<float>(
        {
            {5, 3.5, 10},
            {0, 3, -2},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_elementwise_min") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3.5, 6},
            {0, 0, -5},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 1, 10},
            {-2.5, 3, -2},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_elementwise_min(lhs, rhs, cpu_allocator);

    GenericTensorAccessorW correct = create_2d_accessor_w_with_contents<float>(
        {
            {3, 1, 6},
            {-2.5, 0, -5},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}
