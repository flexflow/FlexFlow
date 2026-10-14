#include "kernels/tensor_accessor_unary_ops.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/repeat_element.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_accessor_exp") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {0, -1, 0.75},
        },
        cpu_allocator);

    GenericTensorAccessorW result = tensor_accessor_exp(input, cpu_allocator);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {std::exp(3.0f), std::exp(3.0f), std::exp(6.0f)},
            {std::exp(0.0f), std::exp(-1.0f), std::exp(0.75f)},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_scale_by_constant") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {0, -1, 0.75},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_scale_by_constant(input, -2.0, cpu_allocator);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {-6, -6, -12},
            {0, 2, -1.5},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_relu") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, -3, -6},
            {0, -1, 0.75},
        },
        cpu_allocator);

    GenericTensorAccessorW result = tensor_accessor_relu(input, cpu_allocator);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {3, 0, 0},
            {0, 0, 0.75},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_scale_by_constant_inplace") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW input = create_2d_accessor_w_with_contents<float>(
        {
            {3, 3, 6},
            {0, -1, 0.75},
        },
        cpu_allocator);

    tensor_accessor_scale_by_constant_inplace(input, -2.0);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {-6, -6, -12},
            {0, 2, -1.5},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(input, correct),
                  check_kv("result", format_accessor_w_contents(input)));
  }

  TEST_CASE("tensor_accessor_broadcast") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3},
            {-0.5},
            {6},
        },
        cpu_allocator);

    TensorDims output_dims = TensorDims{
        FFOrdered{4_p, 1_p, 3_p, 2_p},
    };

    GenericTensorAccessorW result =
        tensor_accessor_broadcast(input, output_dims, cpu_allocator);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        repeat_element(4_n,
                       std::vector<std::vector<std::vector<float>>>{
                           std::vector<std::vector<float>>{
                               repeat_element<float>(2_n, 3.0),
                               repeat_element<float>(2_n, -0.5),
                               repeat_element<float>(2_n, 6.0),
                           }}),
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("input", format_accessor_r_contents(input)),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_transpose") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {0, -1, 0.75},
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_transpose(input, cpu_allocator);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {3, 0},
            {3, -1},
            {6, 0.75},
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_batch_transpose") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
        {
            {
                {3, 3, 6},
                {0, -1, 0.75},
            },
            {
                {5, 1, -2},
                {1, 2, 0},
            },
        },
        cpu_allocator);

    GenericTensorAccessorW result =
        tensor_accessor_batch_transpose(input, cpu_allocator);

    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<float>(
        {
            {
                {3, 0},
                {3, -1},
                {6, 0.75},
            },
            {
                {5, 1},
                {1, 2},
                {-2, 0},
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("tensor_accessor_reduce") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {0, -1, 0.75},
        },
        cpu_allocator);

    SUBCASE("inner dim") {
      GenericTensorAccessorW result =
          tensor_accessor_reduce(input, ff_dim_t{1_n}, cpu_allocator);

      GenericTensorAccessorR correct =
          create_1d_accessor_r_with_contents<float>(
              {
                  3 + 3 + 6,
                  0 + (-1) + 0.75,
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("outer_dim") {
      GenericTensorAccessorW result =
          tensor_accessor_reduce(input, ff_dim_t{0_n}, cpu_allocator);

      GenericTensorAccessorR correct =
          create_1d_accessor_r_with_contents<float>(
              {(3 + 0), (3 + (-1)), (6 + 0.75)}, cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }
}
