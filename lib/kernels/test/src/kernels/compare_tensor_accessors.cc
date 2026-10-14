#include "kernels/compare_tensor_accessors.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compare_tensor_accessors_lt") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_3d_accessor_r_with_contents<float>(
        {
            {
                {1, 3, 2},
                {4, 2, 1},
            },
            {
                {3, 3, 6},
                {2, 1, 5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_3d_accessor_r_with_contents<float>(
        {
            {
                {2, 3, 3},
                {5, 1, 0},
            },
            {
                {1, 5, 4},
                {2, 1, 5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        compare_tensor_accessors_lt(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<bool>(
        {
            {
                {true, false, true},
                {true, false, false},
            },
            {
                {false, true, false},
                {false, false, false},
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("compare_tensor_accessors_le") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_3d_accessor_r_with_contents<float>(
        {
            {
                {4, 2, 1},
            },
            {
                {2, 1, 5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_3d_accessor_r_with_contents<float>(
        {
            {
                {5, 1, 0},
            },
            {
                {2, 1, 5},
            },
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        compare_tensor_accessors_le(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<bool>(
        {
            {
                {true, false, false},
            },
            {
                {true, true, true},
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("compare_tensor_accessors_gt") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {4, 2, 1},
            {2, 1, 5},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 1, 0},
            {2, 1, 5},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        compare_tensor_accessors_gt(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<bool>(
        {
            {false, true, true},
            {false, false, false},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("compare_tensor_accessors_ge") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {4, 2},
            {2, 5},
            {1, 8},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 1},
            {3, 6},
            {1, 0},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        compare_tensor_accessors_ge(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<bool>(
        {
            {false, true},
            {false, false},
            {true, true},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("compare_tensor_accessors_eq") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {4, 2},
            {1, 8},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 2},
            {1, 8},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        compare_tensor_accessors_eq(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<bool>(
        {
            {false, true},
            {true, true},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("compare_tensor_accessors_ne") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR lhs = create_2d_accessor_r_with_contents<float>(
        {
            {4, 2},
            {1, 8},
            {1, 2},
        },
        cpu_allocator);

    GenericTensorAccessorR rhs = create_2d_accessor_r_with_contents<float>(
        {
            {5, 2},
            {1, 8},
            {2, 2},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        compare_tensor_accessors_ne(lhs, rhs, cpu_allocator);
    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<bool>(
        {
            {true, false},
            {false, false},
            {true, false},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}
